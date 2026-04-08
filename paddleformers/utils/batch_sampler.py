#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division, print_function

import math

import numpy as np
import paddle

__all__ = ["MappingBatchSampler", "MappingDistributedBatchSampler", "DistributedBatchSampler"]


class RandomSamplerWithSeed(paddle.io.RandomSampler):
    def __init__(self, data_source, replacement=False, num_samples=None, generator=None) -> None:
        super().__init__(data_source, replacement=replacement, num_samples=num_samples, generator=generator)
        self.epoch = 0

    def set_epoch(self, epoch=0):
        self.epoch = epoch

    def __iter__(self):
        n = len(self.data_source)
        num_samples = self.num_samples if self.num_samples is not None else n
        if self.generator:
            for i in range(num_samples):
                try:
                    index = next(self.generator)
                except StopIteration:
                    return
                yield index
        else:
            for index in (
                np.random.RandomState(self.epoch).choice(np.arange(n), num_samples, replace=self.replacement).tolist()
            ):
                yield index


class MappingBatchSampler(paddle.io.BatchSampler):
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False, consumed_samples=0):

        if shuffle:
            sampler = RandomSamplerWithSeed(dataset)
        else:
            sampler = paddle.io.SequenceSampler(dataset)

        super().__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)
        self.consumed_samples = consumed_samples

    def set_epoch(self, epoch=0, consumed_samples=0):
        self.epoch = epoch
        self.consumed_samples = consumed_samples
        if isinstance(self.sampler, RandomSamplerWithSeed):
            self.sampler.set_epoch(epoch=epoch)

    def __iter__(self):

        # Yield in batches
        local_batch_size = self.batch_size * self._acc_steps
        batch_indices = []
        for idx in self.sampler:
            # Skip consumed samples for resume
            if self.consumed_samples > 0:
                self.consumed_samples -= 1
                continue
            batch_indices.append(idx)
            if len(batch_indices) == local_batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices


class MappingDistributedBatchSampler(paddle.io.DistributedBatchSampler):
    def __init__(
        self, dataset, batch_size, num_replicas=None, rank=None, shuffle=False, drop_last=False, consumed_samples=0
    ):
        super().__init__(
            dataset, batch_size, num_replicas=num_replicas, rank=rank, shuffle=shuffle, drop_last=drop_last
        )
        self.consumed_samples = consumed_samples

    def set_epoch(self, epoch=0, consumed_samples=0):
        self.epoch = epoch
        self.consumed_samples = consumed_samples

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = np.arange(num_samples).tolist()

        # Add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        assert len(indices) == self.total_size

        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(indices)
            self.epoch += 1

        # Subsample for local rank
        def _get_indices_by_batch_size(indices):
            subsampled_indices = []
            last_batch_size = self.total_size % (self.batch_size * self.nranks)
            assert last_batch_size % self.nranks == 0
            last_local_batch_size = last_batch_size // self.nranks

            for i in range(
                self.local_rank * self.batch_size,
                len(indices) - last_batch_size,
                self.batch_size * self.nranks,
            ):
                subsampled_indices.extend(indices[i : i + self.batch_size])

            indices = indices[len(indices) - last_batch_size :]
            subsampled_indices.extend(
                indices[self.local_rank * last_local_batch_size : (self.local_rank + 1) * last_local_batch_size]
            )
            return subsampled_indices

        if self.nranks > 1:
            indices = _get_indices_by_batch_size(indices)

        assert len(indices) == self.num_samples

        assert (
            self.consumed_samples % self.nranks == 0
        ), "The consumed_samples should be divided by nranks. consumed_samples=%d, nranks=%s" % (
            self.consumed_samples,
            self.nranks,
        )

        # Skip consumed samples for resume (per-rank)
        consumed_per_rank = self.consumed_samples // self.nranks
        indices = indices[consumed_per_rank:]

        # Yield in batches
        local_batch_size = self.batch_size * self._acc_steps
        batch_indices = []
        for idx in indices:
            batch_indices.append(idx)
            if len(batch_indices) == local_batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices


class DistributedBatchSampler(paddle.io.BatchSampler):
    """Sampler that restricts data loading to a subset of the dataset.

    Uses interleaved sharding (indices[rank::world_size]) to align with
    ms-swift's BatchSamplerShard, instead of the contiguous-block strategy
    used by paddle.io.DistributedBatchSampler.

    Supports consumed_samples for checkpoint-resume (resume-from-middle-of-epoch).

    Args:
        dataset(paddle.io.Dataset): dataset to sample from.
        batch_size(int): sample indices number in a mini-batch.
        num_replicas(int, optional): number of processes. Defaults to ParallelEnv().nranks.
        rank(int, optional): rank of current process. Defaults to ParallelEnv().local_rank.
        shuffle(bool): whether to shuffle. Uses paddle.randperm with seed=base_seed+epoch.
        drop_last(bool): whether to drop the last incomplete batch.
        consumed_samples(int): total number of samples already consumed across all ranks
            (used to resume from a checkpoint mid-epoch).
        data_seed(int): base random seed. Actual seed per epoch = data_seed + epoch.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas=None,
        rank=None,
        shuffle=False,
        drop_last=False,
        consumed_samples=0,
        data_seed=0,
    ):
        self.dataset = dataset

        assert isinstance(batch_size, int) and batch_size > 0, "batch_size should be a positive integer"
        self.batch_size = batch_size
        assert isinstance(shuffle, bool), "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), "drop_last should be a boolean number"
        self.drop_last = drop_last

        from paddle.distributed import ParallelEnv

        if num_replicas is not None:
            assert isinstance(num_replicas, int) and num_replicas > 0, "num_replicas should be a positive integer"
            self.nranks = num_replicas
        else:
            self.nranks = ParallelEnv().nranks

        if rank is not None:
            assert isinstance(rank, int) and rank >= 0, "rank should be a non-negative integer"
            self.local_rank = rank
        else:
            self.local_rank = ParallelEnv().local_rank

        self.consumed_samples = consumed_samples
        self.base_seed = data_seed
        self.curr_seed = data_seed
        self.epoch = 0

        if self.dataset is None:
            self.num_samples = 0
        else:
            # floor truncation: drop remainder so every rank gets the same count
            total_size = (len(self.dataset) // self.nranks) * self.nranks
            self.num_samples = total_size // self.nranks
        self.total_size = self.num_samples * self.nranks

    def __iter__(self):
        assert (
            self.consumed_samples % self.nranks == 0
        ), "consumed_samples should be divisible by nranks. consumed_samples=%d, nranks=%d" % (
            self.consumed_samples,
            self.nranks,
        )

        # floor truncation — no padding, consistent with ms-swift BatchSamplerShard
        total_size = (len(self.dataset) // self.nranks) * self.nranks

        if self.shuffle:
            # Use paddle.randperm with a deterministic per-epoch seed (base_seed + epoch),
            # consistent with ms-swift: generator.manual_seed(curr_seed)
            paddle.seed(self.curr_seed)
            total_idx = paddle.randperm(total_size).tolist()
            indices = total_idx[self.local_rank :: self.nranks]  # interleaved shard
        else:
            indices = list(range(self.local_rank, total_size, self.nranks))  # interleaved shard

        # Resume from checkpoint: skip already-consumed samples for this rank
        consumed_per_rank = self.consumed_samples // self.nranks
        indices = indices[consumed_per_rank:]

        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if not self.drop_last and len(batch) > 0:
            yield batch

    def __len__(self):
        total_size = (len(self.dataset) // self.nranks) * self.nranks
        per_rank = total_size // self.nranks
        consumed_per_rank = self.consumed_samples // self.nranks
        remaining = per_rank - consumed_per_rank
        if self.drop_last:
            return remaining // self.batch_size
        else:
            return (remaining + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch=0, consumed_samples=0):
        """
        Update epoch and consumed_samples.

        When shuffle=True, the seed for the next iteration will be base_seed + epoch,
        consistent with ms-swift's BatchSamplerShard.set_epoch().

        Args:
            epoch(int): current epoch number.
            consumed_samples(int): total samples consumed across all ranks so far.
                Used to resume from a checkpoint mid-epoch.
        """
        self.curr_seed = self.base_seed + epoch
        self.epoch = epoch
        self.consumed_samples = consumed_samples
