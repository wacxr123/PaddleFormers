# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import hashlib
import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, List, Literal, Optional

import numpy as np
from paddle.io import Dataset, IterableDataset

from paddleformers.datasets.data_utils import (
    get_worker_sliced_iterator,
    pack_by_length,
    postprocess_fc_sequence,
    print_debug_info,
)
from paddleformers.datasets.reader.mix_datasets import create_dataset_instance
from paddleformers.datasets.reader.multi_source_datasets import MultiSourceDataset
from paddleformers.transformers.tokenizer_utils import PretrainedTokenizer
from paddleformers.utils.env import NONE_CHAT_TEMPLATE
from paddleformers.utils.log import logger


@dataclass
class TextSequence:
    """Encapsulated text sequence class."""

    token_ids: List[int]
    position_ids: List[int]
    labels: List[int]
    num_examples: int


@dataclass
class Sequence:
    """Encapsulated sequence class."""

    token_ids: List[int]
    position_ids: List[int]
    labels: List[int]
    num_examples: int
    images: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    audios: List[str] = field(default_factory=list)
    mm_inputs: Dict = field(default_factory=dict)


class BaseSFTDataset:
    def __init__(self, **dataset_config):

        # parameter init
        self.tokenizer = dataset_config.get("tokenizer", None)
        self.dataset_num_proc = dataset_config.get("dataset_num_proc", 1)
        if not self.dataset_num_proc:
            self.dataset_num_proc = 1
        logger.info(f"self.dataset_num_proc: {self.dataset_num_proc}")
        self.dataloader_num_workers = dataset_config.get("dataloader_num_workers", 0)
        if self.dataset_num_proc > 1 and self.dataloader_num_workers > 0:
            raise ValueError("dataset_num_proc and dataloader_num_workers can not be set simultaneously now.")
        self.processor = dataset_config.get("processor", None)
        self.max_seq_len = dataset_config.get("max_seq_len", 8192)
        self.template = dataset_config.get("template_instance", None)
        self.template_backend = dataset_config.get("template_backend", "jinja")
        self.use_template = dataset_config.get("use_template", True)
        self.efficient_eos = True if not self.template else getattr(self.template, "efficient_eos", True)
        self.auto_add_bos = True if not self.template else getattr(self.template, "auto_add_bos", False)
        self.split_multi_turn = dataset_config.get("split_multi_turn", False)
        self.encode_one_turn = dataset_config.get("encode_one_turn", True)
        self.is_pretraining = dataset_config.get("is_pretraining", False)
        self.truncation_strategy = dataset_config.get("truncation_strategy", "delete")
        assert self.truncation_strategy in [
            "oral",
            "delete",
            "right",
            "left",
        ], f"truncation_strategy must be in [oral, delete, right, left], but got {self.truncation_strategy}"
        logger.info(f"[dataflow] truncation_strategy: {self.truncation_strategy}")
        self.truncate_packing = dataset_config.get("truncate_packing", True)
        self.is_valid = dataset_config.get("is_valid", False)
        if self.truncate_packing and not self.is_pretraining:
            logger.warning_once("Truncate packing is only valid in pretraining data flow")
        self.packing = dataset_config.get("packing", False)
        self.greedy_intokens = dataset_config.get("greedy_intokens", True)
        self.dtype = dataset_config.get("dtype", None)
        self.binpacking = dataset_config.get("binpacking", False)
        self.packing_interval = dataset_config.get("packing_interval", 128)
        self.packing_batch_size = dataset_config.get("packing_batch_size", 1000)
        if self.is_pretraining and self.packing and self.truncate_packing:
            logger.info("[dataflow] pretrain dataflow using truncate packing.")

        # special token
        self.begin_token = getattr(self.tokenizer.special_tokens_map, "cls_token", "<|begin_of_sentence|>")
        if isinstance(self.tokenizer, PretrainedTokenizer):
            self.begin_token_id = self.tokenizer._convert_token_to_id([self.begin_token])[0]
        else:
            self.begin_token_id = self.tokenizer.convert_tokens_to_ids([self.begin_token])[0]

        # media placeholder token
        self.placeholder_tokens = []
        if self.template and self.template.mm_plugin:
            for tok in [
                self.template.mm_plugin.image_token,
                self.template.mm_plugin.video_token,
                self.template.mm_plugin.audio_token,
            ]:
                if tok:
                    self.placeholder_tokens.append(tok)
        for i, token in enumerate(self.placeholder_tokens):
            if isinstance(token, str):
                if isinstance(self.tokenizer, PretrainedTokenizer):
                    self.placeholder_tokens[i] = self.tokenizer._convert_token_to_id(token)
                else:
                    self.placeholder_tokens[i] = self.tokenizer.convert_tokens_to_ids(token)

        # data loader + multisource dataset mix
        if self.is_valid:
            dataset_config["random_shuffle"] = False
            dataset_config["greedy_intokens"] = False
            multi_source_dataset = MultiSourceDataset(**dataset_config)
            self.mix_datasets = create_dataset_instance(
                "concat",
                multi_source_dataset,
                **dataset_config,
            )
        else:
            multi_source_dataset = MultiSourceDataset(**dataset_config)
            self.mix_datasets = create_dataset_instance(
                dataset_config["mix_strategy"],
                multi_source_dataset,
                **dataset_config,
                reverse=True,
            )

        self.estimate = False
        # The number of valid samples and skipped samples in estimation
        self.unused_samples = 0
        self.used_samples = 0
        # If used_estimate_samples exceeds max_estimate_samples,stop estimating.
        self.used_estimate_samples = 0
        self.max_estimate_samples = 0
        # set max estimate samples
        if not self.is_valid:
            self.max_estimate_samples = len(self.mix_datasets)

        self.last_printed_percent = 0
        self._estimate_start_time = None
        self.enable_dataset_debug = os.getenv("FLAGS_enable_dataset_debug", "false").lower() in ("true", "1", "t")
        self.mem_debug = os.getenv("FLAGS_enable_mem_debug", "false").lower() in ("true", "1", "t")

        self.sep_token_len = 0
        if self.use_template and self.template_backend != "jinja":
            self.sep_token_len = len(self.tokenizer.tokenize(self.template.chat_sep))

        # The flag indicating whether all examples have been iterated
        self.iter_all_examples = False

        # The number of reserved tokens for each dialog
        self.num_reserved_tokens_for_each_dialog = 0
        if self.use_template:
            # add dynamic eos
            suffix_ids = (
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.template.suffix[-1]))
                if self.template_backend == "custom"
                else [self.tokenizer.eos_token_id]
            )
            self.num_reserved_tokens_for_each_dialog += len(suffix_ids)

            # bos token
            self.num_reserved_tokens_for_each_dialog += 1
        logger.info(f"self.num_reserved_tokens_for_each_dialog: {self.num_reserved_tokens_for_each_dialog}")

        if self.is_pretraining and self.packing and self.truncate_packing:
            self._current_processor_func = self._process_pretraining_tokens
        else:
            self._current_processor_func = self._process_sequence

        # Deferred multiprocessing initialization (workers spawned on first use)
        self._workers_started = False

    def _ensure_workers(self):
        """Lazily spawn worker processes when multiprocessing is first needed."""
        if self.dataset_num_proc > 1 and not self._workers_started:
            self.prefetch_size = self.dataset_num_proc * 2
            self._in_queue = mp.Queue(maxsize=self.prefetch_size)
            self._out_queue = mp.Queue(maxsize=self.prefetch_size)
            self.workers = []
            for _ in range(self.dataset_num_proc):
                worker = mp.Process(target=self._worker_loop, daemon=True)
                worker.start()
                self.workers.append(worker)
            self._workers_started = True

    def __len__(self):
        return len(self.mix_datasets)

    def _worker_loop(self):
        """Worker process main loop."""
        while True:
            try:
                i, example, actual_example_num = self._in_queue.get()
                result = None
                try:
                    result = self._current_processor_func(example, actual_example_num)
                except Exception as e:
                    # result remains None, will be counted as unused_samples in _get_processed_data_iterator
                    print(f"Warning: Error processing example in worker, skipping. Error: {str(e)}")
                self._out_queue.put((i, result))
            except Exception:
                break

    def _get_processed_data_iterator(self, dataset_iterator, actual_example_num, processor_func):
        """Get an iterator that yields processed data, using multiprocessing if enabled.

        Args:
            dataset_iterator: Raw data iterator.
            actual_example_num: Number of examples used.
            processor_func: Function to process each example.

        Yields:
            Processed results in order (skips None results).
        """

        def _rss_mb():
            try:
                with open("/proc/self/status") as _f:
                    for _line in _f:
                        if _line.startswith("VmRSS:"):
                            return int(_line.split()[1]) / 1024
            except Exception:
                pass
            return -1

        _log_interval = 200
        _yield_cnt = 0

        if self.dataset_num_proc > 1:
            # Multiprocessing mode
            self._ensure_workers()
            if self.mem_debug:
                print(f"[MemDebug] workers started, RSS={_rss_mb():.0f} MB, " f"num_proc={self.dataset_num_proc}")
            try:
                pending = 0
                send_idx = 0
                recv_idx = 0
                result_buffer = {}  # Buffer for out-of-order results
                total_samples = len(self.mix_datasets)

                # Pre-fill the queue
                for _ in range(self.prefetch_size):
                    if send_idx >= total_samples:
                        break
                    example = next(dataset_iterator)
                    self._in_queue.put((send_idx, example, actual_example_num))
                    send_idx += 1
                    pending += 1

                if self.mem_debug:
                    print(
                        f"[MemDebug] pre-fill done, RSS={_rss_mb():.0f} MB, "
                        f"pending={pending}, in_q~{self._in_queue.qsize()}, "
                        f"out_q~{self._out_queue.qsize()}"
                    )

                # Process data in streaming fashion, maintaining order
                while pending > 0:
                    idx, result = self._out_queue.get()
                    pending -= 1

                    while send_idx < total_samples and pending < self.prefetch_size:
                        example = next(dataset_iterator)
                        self._in_queue.put((send_idx, example, actual_example_num))
                        send_idx += 1
                        pending += 1

                    # Store result in buffer
                    result_buffer[idx] = result

                    # Yield results in order, skip None
                    while recv_idx in result_buffer:
                        res = result_buffer.pop(recv_idx)
                        recv_idx += 1
                        if res is not None:
                            _yield_cnt += 1
                            if self.mem_debug and _yield_cnt % _log_interval == 0:
                                print(
                                    f"[MemDebug] yielded={_yield_cnt}, RSS={_rss_mb():.0f} MB | "
                                    f"pending={pending}, result_buf={len(result_buffer)}, "
                                    f"in_q~{self._in_queue.qsize()}, out_q~{self._out_queue.qsize()}"
                                )
                            yield res
                        else:
                            if self.estimate:
                                self.used_estimate_samples += actual_example_num
                                self.unused_samples += actual_example_num
            finally:
                if self.mem_debug:
                    print(f"[MemDebug] iteration finished, RSS={_rss_mb():.0f} MB, " f"workers kept alive for reuse")
        else:
            # Single process mode
            for _ in range(len(self.mix_datasets)):
                example = next(dataset_iterator)
                try:
                    result = processor_func(example, actual_example_num)
                except Exception as e:
                    print(f"Warning: Error processing example, skipping. Error: {str(e)}")
                    result = None
                if result is not None:
                    _yield_cnt += 1
                    if self.mem_debug and _yield_cnt % _log_interval == 0:
                        print(f"[MemDebug][single] yielded={_yield_cnt}, RSS={_rss_mb():.0f} MB")
                    yield result
                else:
                    if self.estimate:
                        self.unused_samples += actual_example_num
                        self.used_estimate_samples += actual_example_num

    def _process_sequence(self, example, actual_example_num):
        """Process a single example into a sequence."""
        if self.is_pretraining:
            return self._postprocess_pretraining_sequence(example, actual_example_num)
        else:
            return self._postprocess_sequence(example, actual_example_num)

    def _process_pretraining_tokens(self, example, actual_example_num):
        """Process a pretraining example into tokens."""
        return self._encode_pretraining_messages(example["messages"], actual_example_num)

    def _process_sequence_length(self, example, actual_example_num):
        """Process a single example and return only its token length.

        Used by MapSFTDataset._build_packed_idx to reduce IPC overhead
        when multiprocessing — transfers an int instead of a full Sequence.
        """
        seq = self._process_sequence(example, actual_example_num)
        return len(seq.token_ids) if seq is not None else None

    def _get_packing_mode(self) -> str:
        """Determine packing mode string from dataset config."""
        if self.binpacking:
            return "binpacking"
        elif self.greedy_intokens:
            return "greedy"
        return "sequential"

    def _pack_items(self, items, is_finished=True, return_seqs=False):
        """Unified packing logic using pack_by_length."""
        return pack_by_length(
            items=items,
            max_seq_len=self.max_seq_len,
            packing_mode=self._get_packing_mode(),
            is_finished=is_finished,
            packing_batch_size=self.packing_batch_size,
            return_seqs=return_seqs,
        )

    def _generate_sequences(self):

        # prepare epoch data
        batch_sequence, cur_len = [], 0
        dataset_iterator = get_worker_sliced_iterator(self.mix_datasets)
        actual_example_num = 1

        # pre-training:
        # 1. tokenize all the samples in the sampling pool,
        # 2. combine them into one large sample
        # 3. truncate it into multiple new samples based on the max_seq_len.
        if self.is_pretraining and self.packing and self.truncate_packing:
            take_lengths = []
            buffer = []
            data_iter = self._get_processed_data_iterator(
                dataset_iterator, actual_example_num, self._process_pretraining_tokens
            )
            for tokens in data_iter:
                if self.estimate:
                    self.used_samples += actual_example_num

                idx = 0
                tokens_len = len(tokens)

                while idx < tokens_len:
                    remaining = self.max_seq_len + 1 - len(buffer)
                    take = min(remaining, tokens_len - idx)
                    take_lengths.append(take)
                    buffer.extend(tokens[idx : idx + take])
                    idx += take
                    if len(buffer) == self.max_seq_len + 1:
                        # label shift
                        res_tokens = buffer[:-1]
                        res_labels = buffer[1:]
                        take_lengths[-1] -= 1
                        position_ids = [list(range(item)) for item in take_lengths]
                        sequence = Sequence(
                            token_ids=res_tokens,
                            position_ids=position_ids,
                            labels=res_labels,
                            num_examples=actual_example_num,
                        )
                        batch_sequence = [sequence]
                        yield batch_sequence
                        buffer = []
                        take_lengths = []

                if self.estimate:
                    self.used_estimate_samples += actual_example_num
                    self.print_max_steps_estimate_progress()
                    if self.used_estimate_samples >= self.max_estimate_samples:
                        if buffer:
                            # label shift
                            res_tokens = buffer[:-1]
                            res_labels = buffer[1:]
                            take_lengths[-1] -= 1
                            position_ids = [list(range(item)) for item in take_lengths]
                            sequence = Sequence(
                                token_ids=res_tokens,
                                position_ids=position_ids,
                                labels=res_labels,
                                num_examples=actual_example_num,
                            )
                            batch_sequence = [sequence]
                            yield batch_sequence
                        self.used_estimate_samples = 0
                        # Set flag to False and yield empty list to signal the end of estimation
                        self.estimate = False
                        yield []

            if buffer:
                # label shift
                res_tokens = buffer[:-1]
                res_labels = buffer[1:]
                take_lengths[-1] -= 1
                position_ids = [list(range(item)) for item in take_lengths]
                sequence = Sequence(
                    token_ids=res_tokens,
                    position_ids=position_ids,
                    labels=res_labels,
                    num_examples=actual_example_num,
                )
                batch_sequence = [sequence]
                yield batch_sequence
            self.iter_all_examples = True
        else:
            if not self.packing:
                logger.info("Not using packing mode for data iteration.")
                data_iter = self._get_processed_data_iterator(
                    dataset_iterator, actual_example_num, self._process_sequence
                )
                for sequence in data_iter:
                    if self.estimate:
                        self.used_samples += actual_example_num
                    batch_sequence, cur_len = [sequence], len(sequence.token_ids)
                    yield batch_sequence

                    if self.estimate:
                        self.used_estimate_samples += actual_example_num
                        self.print_max_steps_estimate_progress()
                        if self.used_estimate_samples >= self.max_estimate_samples:
                            self.used_estimate_samples = 0
                            # Set flag to False and yield empty list to signal the end of estimation
                            self.estimate = False
                            yield []
                if len(batch_sequence) > 0:
                    yield batch_sequence
                self.iter_all_examples = True
            elif self.packing:
                if self.binpacking:
                    logger.info("Using binpacking mode for data iteration.")
                    data_iter = self._get_processed_data_iterator(
                        dataset_iterator, actual_example_num, self._process_sequence
                    )
                    accumulated_data = []

                    while True:
                        # Collect batch of sequences
                        batch_sequences = []
                        for _ in range(self.packing_interval):
                            try:
                                seq = next(data_iter)
                                if self.estimate:
                                    self.used_samples += 1
                                if seq:
                                    batch_sequences.append((seq, len(seq.token_ids)))
                            except StopIteration:
                                break

                        finished = len(batch_sequences) < self.packing_interval
                        accumulated_data += batch_sequences

                        # Use unified _pack_items
                        packed, accumulated_data = self._pack_items(
                            accumulated_data, is_finished=finished, return_seqs=True
                        )

                        for pack in packed:
                            if len(pack) > 0:
                                yield pack

                        if self.estimate:
                            self.used_estimate_samples += len(batch_sequences)
                            self.print_max_steps_estimate_progress()
                            # Stop estimation if the number of samples used in estimation is larger than max_estimate_samples
                            if self.used_estimate_samples >= self.max_estimate_samples:
                                # Set flag to False and yield empty list to signal the end of estimation
                                self.estimate = False
                                yield []

                        if finished:
                            self.iter_all_examples = True
                            break
                elif self.greedy_intokens:
                    logger.info("Using greedy packing mode for data iteration.")
                    # Pseudo multiple rounds + group greedy intokens.
                    buffer_size = self.packing_interval
                    sequences_buffer = []
                    data_iter = self._get_processed_data_iterator(
                        dataset_iterator, actual_example_num, self._process_sequence
                    )
                    for sequence in data_iter:
                        if self.estimate:
                            self.used_samples += actual_example_num

                        sequences_buffer.append(sequence)

                        if len(sequences_buffer) >= buffer_size:
                            # Use unified _pack_items
                            generate_packs = self._pack_items(sequences_buffer, return_seqs=True)
                            for pack in generate_packs:
                                if len(pack) > 0:
                                    yield pack
                            sequences_buffer = []

                        if self.estimate:
                            self.used_estimate_samples += actual_example_num
                            self.print_max_steps_estimate_progress()
                            # Stop estimation if the number of samples used in estimation is larger than max_estimate_samples
                            if self.used_estimate_samples >= self.max_estimate_samples:
                                # Yield left packs before estimation ends
                                if len(sequences_buffer) > 0:
                                    generate_packs = self._pack_items(sequences_buffer, return_seqs=True)
                                    for pack in generate_packs:
                                        if len(pack) > 0:
                                            yield pack
                                # Set flag to False and yield empty list to signal the end of estimation
                                self.estimate = False
                                yield []

                    if len(sequences_buffer) > 0:
                        generate_packs = self._pack_items(sequences_buffer, return_seqs=True)
                        for pack in generate_packs:
                            if len(pack) > 0:
                                yield pack

                    self.iter_all_examples = True
                else:
                    logger.info("Using base packing mode for data iteration.")
                    # base packing mode
                    data_iter = self._get_processed_data_iterator(
                        dataset_iterator, actual_example_num, self._process_sequence
                    )
                    for sequence in data_iter:
                        if self.estimate:
                            self.used_samples += actual_example_num
                        if cur_len + len(sequence.token_ids) <= self.max_seq_len:
                            batch_sequence.append(sequence)
                            cur_len += len(sequence.token_ids)
                        else:
                            yield batch_sequence
                            batch_sequence, cur_len = [sequence], len(sequence.token_ids)

                        if self.estimate:
                            self.used_estimate_samples += actual_example_num
                            self.print_max_steps_estimate_progress()
                            if self.used_estimate_samples >= self.max_estimate_samples:
                                # Yield left batch sequence before estimation ends
                                if len(batch_sequence) > 0:
                                    yield batch_sequence
                                self.used_estimate_samples = 0
                                # Set flag to False and yield empty list to signal the end of estimation
                                self.estimate = False
                                yield []
                    if len(batch_sequence) > 0:
                        yield batch_sequence
                    self.iter_all_examples = True

    def __iter__(self):
        """
        Rewrite the __iter__ method to implement dataset iteration.
        Each iteration returns a Sequence-type element.
        """
        if self.is_valid:
            yield from self.__iter_func()
        else:
            while True:
                yield from self.__iter_func()

    def _encode_pretraining_messages(self, messages, actual_example_num):
        # tokens
        content = messages[0]["content"]
        tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(content))
        # Add an EOS token at the end of each sample
        tokens = tokens + [self.tokenizer.eos_token_id]
        return tokens

    def _postprocess_pretraining_sequence(self, example, actual_example_num):

        messages = example.get("messages", [])
        images = example.get("images", [])
        videos = example.get("videos", [])
        audios = example.get("audios", [])

        if len(images) == 0 and len(videos) == 0 and len(audios) == 0:
            tokens = self._encode_pretraining_messages(messages, actual_example_num)
            if len(tokens) > self.max_seq_len + 1:
                # Truncate the sequence to the maximum length
                tokens = tokens[: self.max_seq_len + 1]
            res_tokens = tokens[:-1]
            res_labels = tokens[1:]
            pos_ids = list(range(len(res_tokens)))
            sequence = Sequence(
                token_ids=res_tokens,
                position_ids=pos_ids,
                labels=res_labels,
                num_examples=actual_example_num,
            )
            return sequence
        else:
            mm_inputs = self.template.mm_plugin.get_mm_inputs(
                images,
                videos,
                audios,
                self.processor,
                imglens=[len(images)],
                vidlens=[len(videos)],
                audlens=[len(audios)],
                batch_ids=None,
                messages=messages,
            )

            messages = self.template.mm_plugin.process_messages(
                messages, images, videos, audios, mm_inputs, self.processor
            )

            tokens = self._encode_pretraining_messages(messages, actual_example_num)
            if len(tokens) > self.max_seq_len + 1:
                # Truncate the sequence to the maximum length
                tokens = tokens[: self.max_seq_len + 1]

            labels = self.template.mm_plugin.process_tokens(tokens, self.processor)

            # label shift
            labels = labels[1:] + [-100]

            pos_ids = list(range(len(tokens)))  # only pure text, mm_position_ids will be reconstructed in collate.py

            if all(x == -100 for x in labels):
                logger.warning(f"[SKIP] all labels set to 0: {example}")
                return None

            assert len(tokens) == len(labels), f"{len(tokens)}-{len(labels)}"

            if self.enable_dataset_debug:
                logger.info("\n" + "=" * 50)
                logger.info("[dataset debug] Debug mode enabled")
                if hasattr(self, "tokenizer"):
                    print("========================================")
                    print("tokens: ", [tokens])
                    print_debug_info(self.tokenizer, tokens, "input")
                    print("========================================\n")

                    filtered_labels = [x for x in labels if x != -100]  # remove -100
                    print("========================================")
                    print("labels: ", [labels])
                    print_debug_info(self.tokenizer, filtered_labels, "labels")
                    print("========================================\n")
                else:
                    logger.info("[dataset debug] Tokenizer not available")
                logger.info("=" * 50 + "\n")

            return Sequence(
                token_ids=tokens,
                position_ids=pos_ids,
                labels=labels,
                num_examples=actual_example_num,
                images=images,
                videos=videos,
                audios=audios,
                mm_inputs=mm_inputs,
            )

    def _postprocess_sequence(self, example, actual_example_num):
        """Process code completion examples into token sequences.

        Args:
            example: The input example containing code components.
            actual_example_num (int): Number of examples used.

        Returns:
            Sequence: Processed sequence or None if invalid.
        """
        system = example.get("system", None)
        tools = example.get("tools", None)
        images = example.get("images", [])
        videos = example.get("videos", [])
        audios = example.get("audios", [])
        objects = example.get("objects", {})
        mm_inputs = None

        if self.use_template:
            if self.template_backend == "jinja":
                if not self.tokenizer.chat_template:
                    self.tokenizer.chat_template = NONE_CHAT_TEMPLATE
                if self.split_multi_turn:
                    encoded_pairs = postprocess_fc_sequence(self.tokenizer, example)
                else:
                    encoded_pairs = self.tokenizer.encode_chat_inputs(example, encode_one_turn=self.encode_one_turn)
            else:
                messages = self.template.grounding_plugin.process_messages(
                    example["messages"],
                    objects,
                )
                mm_inputs = self.template.mm_plugin.get_mm_inputs(
                    images,
                    videos,
                    audios,
                    self.processor,
                    imglens=[len(images)],
                    vidlens=[len(videos)],
                    audlens=[len(audios)],
                    batch_ids=None,
                    messages=messages,
                    dtype=self.dtype,
                )
                messages = self.template.mm_plugin.process_messages(
                    messages, images, videos, audios, mm_inputs, self.processor
                )
                encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        else:
            encoded_pairs = self.tokenizer.encode_chat_inputs_with_no_template(
                example, encode_one_turn=self.encode_one_turn
            )

        cur_len = self.num_reserved_tokens_for_each_dialog
        tokens_chunks = []
        labels_chunks = []
        accumulated_tokens_len = 0

        for turn_index in range(len(encoded_pairs) - 1, -1, -1):
            tokens_src, tokens_target = encoded_pairs[turn_index]
            if len(tokens_target) == 0:
                logger.warning(f"[SKIP] The length of encoded assistant tokens is 0: {example}")
                return None

            if self.truncation_strategy == "oral":
                remaining_len = self.max_seq_len - cur_len
                if len(tokens_src) + len(tokens_target) > remaining_len:
                    if images or videos or audios:
                        # If there is multimodal data, do not truncate it; just discard it directly.
                        sub_src = example["messages"][0]["content"].strip()[:50]
                        logger.warning(f"[SKIP] This data is too long: {sub_src}...")
                        return None
                    # If the source (src) exceeds length limit, discard this round of conversation data
                    # If the target (tgt) exceeds length limit, truncate it
                    if len(tokens_src) > remaining_len:
                        break
                    else:
                        tokens_target = tokens_target[: remaining_len - len(tokens_src)]

            labels_src = [-100] * len(tokens_src)

            # Perform additional processing on chat sep.
            # If eos is valid, replace it with eos for learning;
            # otherwise, replace it with -100 and do not learn
            if not self.use_template or self.template_backend == "jinja":
                labels_target = tokens_target
            else:
                if turn_index != (len(encoded_pairs) - 1):
                    labels_target = (
                        tokens_target[: len(tokens_target) - self.sep_token_len] + [-100] * self.sep_token_len
                    )
                else:
                    labels_target = tokens_target

            if not example["label"][turn_index]:
                labels_target = [-100] * len(labels_target)

            tokens_chunks.append(tokens_src + tokens_target)
            labels_chunks.append(labels_src + labels_target)

            accumulated_tokens_len += len(tokens_src) + len(tokens_target)
            cur_len = accumulated_tokens_len

        tokens_chunks.reverse()
        labels_chunks.reverse()
        tokens = list(chain.from_iterable(tokens_chunks))
        labels = list(chain.from_iterable(labels_chunks))
        del tokens_chunks, labels_chunks

        # Not even one turn can be added, so need to do warning and skip this example
        if len(tokens) <= self.num_reserved_tokens_for_each_dialog:
            try:
                # For print log
                sub_src = example["messages"][0]["content"].strip()[:50]
                sub_tgt = example["messages"][-1]["content"].strip()[-50:]
                msg = "too short" if len(tokens) > 0 else "too long"
                logger.warning(f"This data is {msg}: '{{'src':[{sub_src}, ……],'tgt':[……{sub_tgt}]}}'")
            except Exception:
                logger.warning("[SKIP] wrong example")
            return None

        if self.use_template:
            # add dynamic eos
            suffix_ids = (
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.template.suffix[-1]))
                if self.template_backend == "custom"
                else [self.tokenizer.eos_token_id]
            )
            self._add_dynamic_eos(tokens, labels, suffix_ids)

            # Maybe left truncated, so need to add begin_token
            if self.auto_add_bos and self.begin_token_id and tokens[0] != self.begin_token_id:
                tokens = [self.begin_token_id] + tokens
                labels = [-100] + labels

            # Add EOS token at the end
            if self.efficient_eos:
                tokens.extend(suffix_ids)
                labels.extend(suffix_ids)

        # data truncate
        if self.truncation_strategy != "oral":
            tokens, labels = self._encode_truncated(tokens, labels)
            if not tokens:
                sub_src = example["messages"][0]["content"].strip()[:50]
                logger.warning(f"[SKIP] data is deleted by truncation strategy: {sub_src}...")
                return None
        else:
            if len(tokens) > self.max_seq_len:
                raise RuntimeError(f"token_ids is too long: {len(tokens)}")

        # label shift
        labels = labels[1:] + [-100]

        pos_ids = list(range(len(tokens)))

        if all(x == -100 for x in labels):
            logger.warning(f"[SKIP] all labels set to -100: {example}")
            return None

        assert len(tokens) == len(labels), f"{len(tokens)}-{len(labels)}"

        if self.enable_dataset_debug:
            logger.info("\n" + "=" * 50)
            logger.info("[dataset debug] Debug mode enabled")
            if hasattr(self, "tokenizer"):
                print("========================================")
                print("tokens: ", [tokens])
                print_debug_info(self.tokenizer, tokens, "input")
                print("========================================\n")

                filtered_labels = [x for x in labels if x != -100]  # remove -100
                print("========================================")
                print("labels: ", [labels])
                print_debug_info(self.tokenizer, filtered_labels, "labels")
                print("========================================\n")
            else:
                logger.info("[dataset debug] Tokenizer not available")
            logger.info("=" * 50 + "\n")

        return Sequence(
            token_ids=tokens,
            position_ids=pos_ids,
            labels=labels,
            num_examples=actual_example_num,
            images=images,
            videos=videos,
            audios=audios,
            mm_inputs=mm_inputs,
        )

    @staticmethod
    def _get_length(input_ids, labels):
        # input_ids might be a tensor.
        lengths = [0]
        if input_ids is not None:
            lengths.append(len(input_ids))
        if labels is not None:
            lengths.append(len(labels))
        length = max(lengths)
        return length

    def _truncate(
        self,
        input_ids: List[int],
        labels: Optional[List[int]],
        truncation_strategy: Literal["left", "right"],
    ):
        max_len = self.max_seq_len
        placeholder_set = set(self.placeholder_tokens)

        is_placeholder = [tok in placeholder_set for tok in input_ids]
        placeholder_idx = [i for i, v in enumerate(is_placeholder) if v]

        if len(placeholder_idx) >= max_len:
            keep_idx = placeholder_idx[:max_len]
        else:
            remain = max_len - len(placeholder_idx)
            non_placeholder_idx = [i for i, v in enumerate(is_placeholder) if not v]

            if truncation_strategy == "left":
                extra_idx = non_placeholder_idx[-remain:]
            else:
                extra_idx = non_placeholder_idx[:remain]

            keep_idx = sorted(placeholder_idx + extra_idx)

        input_ids = [input_ids[i] for i in keep_idx]
        labels = [labels[i] for i in keep_idx]

        return input_ids, labels

    def _encode_truncated(self, input_ids, labels):
        length = self._get_length(input_ids, labels)
        if self.max_seq_len is not None and length > self.max_seq_len:
            if self.truncation_strategy == "delete":
                return None, None
            if self.truncation_strategy in {"right", "left"}:
                input_ids, labels = self._truncate(input_ids, labels, truncation_strategy=self.truncation_strategy)
        return input_ids, labels

    def print_max_steps_estimate_progress(self):
        current_percent = (self.used_estimate_samples / self.max_estimate_samples) * 100
        if self._estimate_start_time is None:
            self._estimate_start_time = time.time()
        # Print progress at every 5% interval.
        if int(current_percent) // 5 > self.last_printed_percent // 5:
            elapsed = time.time() - self._estimate_start_time
            print(f"[Estimate Max Steps Progress]: {current_percent:.0f}% (elapsed: {elapsed:.1f}s)")
            self.last_printed_percent = current_percent

    @staticmethod
    def _add_dynamic_eos(input_ids, labels, suffix_tokens_id):
        # Adapted from:
        # https://github.com/modelscope/ms-swift
        # Original author: modelscope
        # License: Apache-2.0
        suffix_len = len(suffix_tokens_id)
        start = 0
        for i in range(1, len(labels) + 1):
            if labels[i - 1] >= 0 and i < len(labels) and labels[i] == -100:
                start = i
            elif start > 0 and labels[i - 1] == -100 and (i == len(labels) or labels[i] >= 0):
                # [0, 1, 2, -100(start), -100, 3(i), 4]
                length = i - start
                if length >= suffix_len and input_ids[start : start + suffix_len] == suffix_tokens_id:
                    labels[start : start + suffix_len] = suffix_tokens_id


class IteratorSFTDataset(BaseSFTDataset, IterableDataset):
    def __init__(self, **dataset_config):
        super().__init__(**dataset_config)

    def __iter__(self):
        if self.is_valid:
            yield from self._generate_sequences()
        else:
            while True:
                yield from self._generate_sequences()


class MapSFTDataset(BaseSFTDataset, Dataset):
    @staticmethod
    def _serialize_packed_idx(packed_idx: List[List[int]]):
        """Serialize packed_idx to CSR format (data + offsets arrays)."""
        offsets = np.zeros(len(packed_idx) + 1, dtype=np.int32)
        for i, pack in enumerate(packed_idx):
            offsets[i + 1] = offsets[i] + len(pack)
        data = np.empty(int(offsets[-1]), dtype=np.int32)
        pos = 0
        for pack in packed_idx:
            n = len(pack)
            data[pos : pos + n] = pack
            pos += n
        return data, offsets

    @staticmethod
    def _deserialize_packed_idx(data, offsets) -> List[List[int]]:
        """Reconstruct packed_idx from CSR format arrays."""
        return [data[offsets[i] : offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]

    def __init__(self, **dataset_config):
        super().__init__(**dataset_config)

        self._dataset_config = dataset_config  # preserved for cache key
        self.packed_idx_cache_dir: Optional[str] = dataset_config.get("packed_idx_cache_dir", None)

        # Always store raw data for index-based access
        self.raw_data = list(self.mix_datasets)

        if self.packing:
            # Use length-only processor for packing index building
            self._current_processor_func = self._process_sequence_length
            self._build_packed_idx()
        else:
            logger.info(f"[MapSFTDataset] packing=False, total samples: {len(self.raw_data)}")
            self.n_try_fetch = min(10, len(self.raw_data))
            self.random_state = np.random.RandomState(None)
            self.traceback_limit = 10
            self._traceback_counter = 0
            self._idx = 0
            self._idx_list = self.random_state.permutation(len(self.raw_data)).tolist()

    def _build_packed_idx(self):
        """First pass: tokenize all samples once to get lengths, then pack into index lists.

        Supports multiprocessing via dataset_num_proc for the tokenization pass.
        Uses pack_by_length for grouping, supporting binpacking, greedy_intokens,
        and sequential packing strategies.

        Stores only packed_idx (List[List[int]]) instead of full token tensors,
        reducing memory from O(N * seq_len tokens) to O(N * seq_len chars).
        """
        from tqdm import tqdm

        # Try loading from cache first
        if self.packed_idx_cache_dir is not None:
            cached = self._load_packed_idx_cache()
            if cached is not None:
                self.packed_idx = cached
                return

        logger.info("[MapSFTDataset] packing=True, building packed index (lazy storage)...")
        actual_example_num = 1

        # Collect (raw_data_idx, token_length) for valid samples, skip invalid ones
        idx_len_pairs = []  # [(raw_idx, token_len), ...]

        if self.dataset_num_proc > 1:
            # Multiprocessing path: reuse existing worker pool infrastructure
            self._ensure_workers()
            pending, send_idx, recv_idx = 0, 0, 0
            result_buffer = {}
            total = len(self.raw_data)

            # Pre-fill the queue
            for _ in range(min(self.prefetch_size, total)):
                self._in_queue.put((send_idx, self.raw_data[send_idx], actual_example_num))
                send_idx += 1
                pending += 1

            with tqdm(total=total, desc="[MapSFTDataset] Tokenizing for lengths") as pbar:
                while pending > 0:
                    idx, result = self._out_queue.get()
                    pending -= 1
                    pbar.update(1)

                    # Refill the queue
                    while send_idx < total and pending < self.prefetch_size:
                        self._in_queue.put((send_idx, self.raw_data[send_idx], actual_example_num))
                        send_idx += 1
                        pending += 1

                    # Buffer results for in-order processing
                    result_buffer[idx] = result

                    # Yield results in order
                    while recv_idx in result_buffer:
                        res = result_buffer.pop(recv_idx)
                        if res is not None:
                            idx_len_pairs.append((recv_idx, res))
                        recv_idx += 1
        else:
            # Single-threaded path
            for raw_idx, example in enumerate(tqdm(self.raw_data, desc="[MapSFTDataset] Tokenizing for lengths")):
                try:
                    length = self._process_sequence_length(example, actual_example_num)
                    if length is not None:
                        idx_len_pairs.append((raw_idx, length))
                except Exception as e:
                    logger.warning(f"[MapSFTDataset] Skipping example {raw_idx}: {e}")

        logger.info(f"[MapSFTDataset] Valid samples: {len(idx_len_pairs)} / {len(self.raw_data)}")

        self.packed_idx = self._pack_items(idx_len_pairs, is_finished=True, return_seqs=False)

        logger.info(f"[MapSFTDataset] packing=True, total packs: {len(self.packed_idx)}")

        # Save cache after successful build
        if self.packed_idx_cache_dir is not None:
            self._save_packed_idx_cache()

    def _compute_cache_key(self) -> str:
        """Compute a 16-char SHA-256 digest of all parameters that affect packed_idx."""
        cfg = self._dataset_config
        tokenizer_id = getattr(self.tokenizer, "name_or_path", None) or type(self.tokenizer).__name__
        template_id = type(self.template).__name__ if self.template else "NoTemplate"
        key_dict = {
            "split": str(cfg.get("split", "")),
            "task_group": str(cfg.get("task_group", "")),
            "task_group_prob": str(cfg.get("task_group_prob", "")),
            "sub_dataset_type": str(cfg.get("sub_dataset_type", "")),
            "random_seed": str(cfg.get("random_seed", 0)),
            "random_shuffle": str(cfg.get("random_shuffle", True)),
            "num_samples_each_epoch": str(cfg.get("num_samples_each_epoch", 0)),
            "max_seq_len": str(self.max_seq_len),
            "tokenizer": tokenizer_id,
            "template": template_id,
            "template_backend": self.template_backend,
            "use_template": str(self.use_template),
            "split_multi_turn": str(self.split_multi_turn),
            "encode_one_turn": str(self.encode_one_turn),
            "is_pretraining": str(self.is_pretraining),
            "binpacking": str(self.binpacking),
            "greedy_intokens": str(self.greedy_intokens),
            "packing_interval": str(self.packing_interval),
            "packing_batch_size": str(self.packing_batch_size),
        }
        key_str = json.dumps(key_dict, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:16]

    def _save_packed_idx_cache(self) -> None:
        """Serialize packed_idx to .npz and write .meta.json atomically."""
        cache_key = self._compute_cache_key()
        data_path = os.path.join(self.packed_idx_cache_dir, f"packed_idx_{cache_key}")
        meta_path = os.path.join(self.packed_idx_cache_dir, f"packed_idx_{cache_key}.meta.json")

        try:
            os.makedirs(self.packed_idx_cache_dir, exist_ok=True)

            data_arr, offsets_arr = self._serialize_packed_idx(self.packed_idx)
            np.savez_compressed(data_path, data=data_arr, offsets=offsets_arr)

            meta = {
                "hash": cache_key,
                "created_at": datetime.datetime.now().astimezone().isoformat(),
                "num_packs": len(self.packed_idx),
                "num_samples": sum(len(p) for p in self.packed_idx),
                "max_seq_len": self.max_seq_len,
                "packing_mode": self._get_packing_mode(),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            logger.info(f"[MapSFTDataset] Saved packed_idx cache to {data_path}")
        except Exception as e:
            logger.warning(f"[MapSFTDataset] Failed to save packed_idx cache: {e}")
            for p in [data_path, meta_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    def _load_packed_idx_cache(self) -> Optional[List[List[int]]]:
        """Try to load packed_idx from cache. Returns None on any miss or error (silent fallback)."""
        cache_key = self._compute_cache_key()
        data_path = os.path.join(self.packed_idx_cache_dir, f"packed_idx_{cache_key}.npz")
        meta_path = os.path.join(self.packed_idx_cache_dir, f"packed_idx_{cache_key}.meta.json")

        if not os.path.exists(data_path) or not os.path.exists(meta_path):
            logger.info("[MapSFTDataset] packed_idx cache not found, will build from scratch.")
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if meta.get("hash") != cache_key:
                logger.warning("[MapSFTDataset] Cache hash mismatch, ignoring cache.")
                return None

            npz = np.load(data_path)
            packed_idx = self._deserialize_packed_idx(npz["data"], npz["offsets"])
            logger.info(
                f"[MapSFTDataset] Loaded packed_idx cache from {data_path}, "
                f"skip building... ({len(packed_idx)} packs)"
            )
            return packed_idx
        except Exception as e:
            logger.warning(f"[MapSFTDataset] Failed to load packed_idx cache (will rebuild): {e}")
            return None

    def __len__(self):
        if self.packing:
            return len(self.packed_idx)
        return len(self.raw_data)

    def __getitem__(self, idx):
        if self.packing:
            # Second tokenize: re-process each raw sample in the pack on demand
            actual_example_num = 1
            sequences = []
            for raw_idx in self.packed_idx[idx]:
                example = self.raw_data[raw_idx]
                try:
                    seq = self._process_sequence(example, actual_example_num)
                    if seq is not None:
                        sequences.append(seq)
                except Exception as e:
                    logger.warning(f"[MapSFTDataset] __getitem__ skipping raw_idx={raw_idx}: {e}")
            return sequences

        actual_example_num = 1

        for i in range(self.n_try_fetch):
            if i == 0:
                current_idx = idx
            else:
                current_idx = self._idx_list[self._idx]
                self._idx = (self._idx + 1) % len(self.raw_data)

            example = self.raw_data[current_idx]
            try:
                sequence = self._process_sequence(example, actual_example_num)

                if sequence is not None:
                    return [sequence]

                # sequence is None, try next
                if self.traceback_limit is not None and self._traceback_counter < self.traceback_limit:
                    logger.warning(
                        f"[MapSFTDataset] Example at index {current_idx} returned None, "
                        "another piece of data will be randomly selected."
                    )
                    self._traceback_counter += 1

            except Exception:
                if self.traceback_limit is not None and self._traceback_counter < self.traceback_limit:
                    import traceback

                    logger.info(traceback.format_exc())
                    logger.warning(
                        "[MapSFTDataset] There are errors in data processing, "
                        "another piece of data will be randomly selected."
                    )
                    self._traceback_counter += 1

        raise ValueError(
            f"[MapSFTDataset] Failed to retrieve valid data after {self.n_try_fetch} attempts. "
            "You can avoid this issue by checking your data quality."
        )
