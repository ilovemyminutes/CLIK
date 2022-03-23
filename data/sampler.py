import random
import warnings
from copy import deepcopy

import numpy as np
from torch.utils.data import Sampler

from data import PreferenceDiscrimDataset, SemanticMatchingDataset


class SemanticMatchingBatchSampler(Sampler):
    """Context Matching Task를 위한 Batch Sampler
    Context Matching Task: 기획전 텍스트 정보와 해당 기획전의 상품 이미지 간 contrastive learning

    Args:
        dataset (SemanticMatchingDataset)
        batch_size (int)
        drop_last (bool)
        num_steps (int): 1 에폭 당 step 수를 결정(default: None)
          - None: len(dataset), batch_size, drop_last를 고려하여 step 수를 측정
          - others: Preference discrimination task와 step 수를 동기화해야 할 경우 직접 step 수를 입력

    Examples:
        # matching task만을 활용할 경우
        matchingset = SemanticMatchingDataset(meta, img_dir, ...)
        matchingsampler = SemanticMatchingBatchSampler(matchingset)
        matchingloader = DataLoader(matchingset, batch_sampler=matchingsampler)

        # discrimination task과 함께 활용할 경우
        discrimset = PreferenceDiscrimDataset(meta, img_dir, ...)
        discrimsampler = PreferenceDiscrimBatchSampler(discrimset)
        discrimloader = Dataloader(discrimset, batch_sampler=discrimsampler)

        num_steps = len(discrimset)
        matchingset = SemanticMatchingDataset(meta, img_dir, ...)
        matchingsampler = SemanticMatchingBatchSampler(matchingset, num_steps=num_steps)
        matchingloader = DataLoader(matchingset, batch_sampler=matchingsampler)
    """

    def __init__(
        self,
        dataset: SemanticMatchingDataset,
        matching_size: int = 128,
        drop_last: bool = True,
        num_steps: int = None,  # step 수를 직접 정할 수 있음
        seed: int = None,
    ):
        if len(dataset.unique_plan_ids) < matching_size:
            warnings.warn(
                f"'matching_size({matching_size})' is greater than the number of plans({len(dataset.unique_plan_ids)}). 'matching_size' will be modified: ({matching_size}) -> ({len(dataset.unique_plan_ids)})"
            )
            matching_size = len(dataset.unique_plan_ids)

        self.dataset = dataset
        self.batch_size = matching_size
        self.drop_last = drop_last
        self.seed = seed

        if num_steps is not None:
            self.drop_last = True
            self.last_batch_size = None
            self.num_steps = num_steps

        elif drop_last:
            self.last_batch_size = None
            self.num_steps = len(dataset) // matching_size

        else:
            self.last_batch_size = len(dataset) % matching_size
            self.num_steps = (
                len(dataset) // matching_size
                if self.last_batch_size == 0
                else len(dataset) // matching_size + 1
            )

    def __iter__(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        batches = []
        for step in range(self.num_steps):
            if (not self.drop_last) and (step == self.num_steps - 1):
                replace = (
                    False
                    if len(self.dataset.unique_plan_ids) >= self.last_batch_size
                    else True
                )
                plan_group = np.random.choice(
                    self.dataset.unique_plan_ids,
                    self.last_batch_size,
                    replace=replace,
                )
                assert len(plan_group) == len(
                    set(plan_group)
                ), f"len(plan_group), {len(plan_group)}, len(set(plan_group), {len(set(plan_group))}, batch_size, {self.last_batch_size}"
            else:
                replace = (
                    False
                    if len(self.dataset.unique_plan_ids) >= self.batch_size
                    else True
                )
                plan_group = np.random.choice(
                    self.dataset.unique_plan_ids, self.batch_size, replace=replace
                )
                assert len(plan_group) == len(
                    set(plan_group)
                ), f"len(plan_group), {len(plan_group)}, len(set(plan_group), {len(set(plan_group))}, batch_size, {self.batch_size}"
            batches.append(plan_group)
        return iter(batches)

    def __len__(self):
        return self.num_steps


class PreferenceDiscrimBatchSampler(Sampler):
    """Preference Discrimination Task를 위한 Batch Sampler
    Preference Discrimination Task: 기획전 상품 중 CTR이 가장 높을 만 한 이미지를 추출하는 metric learning

    Args:
        dataset (PreferenceDiscrimBatchSampler)
        sampling_iter: 매 step마다 샘플링할 횟수
          - one_step_one_plan=True: 하나의 기획전으로부터 sampling_iter번 만큼 샘플링을 수행
          - one_step_one_plan=False: sampling_iter개의 기획전으로부터 1번씩 샘플링을 수행
        replace:
          - False: 모든 iterating 동안 겹치는 기획전이 없도록 샘플링을 수행

    Examples:
        # discrimination task과 함께 활용할 경우
        discrimset = PreferenceDiscrimDataset(meta, img_dir, ...)
        discrimsampler = PreferenceDiscrimBatchSampler(discrimset)
        discrimloader = Dataloader(discrimset, batch_sampler=discrimsampler)
    """

    def __init__(
        self,
        dataset: PreferenceDiscrimDataset,
        sampling_iter: int = 3,
        one_step_one_plan: bool = True,
        seed: int = None,
    ):
        if not one_step_one_plan and len(dataset.unique_plan_ids) < sampling_iter:
            raise ValueError(
                f"There're {len(dataset.unique_plan_ids)} plans in meta data that is short to sampling with 'replace=False'"
            )
        self.dataset = dataset
        self.sampling_iter = sampling_iter
        self.one_step_one_plan = one_step_one_plan
        self.replace = (
            False
            if not one_step_one_plan and len(dataset.unique_plan_ids) >= sampling_iter
            else True
        )
        self.num_steps = (
            len(dataset.unique_plan_ids)
            if self.replace or one_step_one_plan
            else len(dataset.unique_plan_ids) // sampling_iter
        )
        self.seed = seed

    def __iter__(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.one_step_one_plan:
            batches = np.random.choice(
                self.dataset.unique_plan_ids, self.num_steps, replace=False
            )[:, np.newaxis]
            batches = np.repeat(batches, self.sampling_iter, axis=1).tolist()

        else:
            batches = []
            if self.replace:
                for _ in range(self.num_steps):
                    batch = np.random.choice(
                        self.dataset.unique_plan_ids, self.sampling_iter, replace=False
                    ).tolist()
                    batches.append(batch)

            else:
                shuffled_plan_ids = deepcopy(self.dataset.unique_plan_ids)
                random.shuffle(shuffled_plan_ids)

                for _ in range(self.num_steps):
                    batch = [shuffled_plan_ids.pop() for _ in range(self.sampling_iter)]
                    assert len(batch) == len(set(batch))
                    batches.append(batch)

        return iter(batches)

    def __len__(self):
        return self.num_steps
