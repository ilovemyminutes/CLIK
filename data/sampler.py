import random
from copy import deepcopy
import warnings
import numpy as np
from torch.utils.data import Sampler
from data import TopicMatchingDataset, ImageRankingDataset


class TopicMatchingBatchSampler(Sampler):
    """Context Matching Task를 위한 Batch Sampler
    Topic Matching Task: 기획전 텍스트 정보와 해당 기획전의 상품 이미지 간 contrastive learning

    Args:
        dataset (TopicMatchingDataset)
        batch_size (int)
        drop_last (bool)
        num_steps (int): 1 에폭 당 step 수를 결정(default: None)
          - None: len(dataset), batch_size, drop_last를 고려하여 step 수를 측정
          - others: Image Ranking task와 step 수를 동기화해야 할 경우 직접 step 수를 입력

    Examples:
        # Topic Matching task만을 활용할 경우
        matching_set = TopicMatchingDataset(meta, img_dir, ...)
        matching_sampler = TopicMatchingBatchSampler(matching_set)
        matching_loader = DataLoader(matching_set, batch_sampler=matching_sampler)

        # Image Ranking task과 함께 활용할 경우
        ranking_set = ImageRankingDataset(meta, img_dir, ...)
        ranking_sampler = ImageRankingBatchSampler(ranking_set)
        ranking_loader = Dataloader(ranking_set, batch_sampler=ranking_sampler)

        num_steps = len(ranking_set)
        matching_set = TopicMatchingDataset(meta, img_dir, ...)
        matching_sampler = TopicMatchingBatchSampler(matching_set, num_steps=num_steps)
        matching_loader = DataLoader(matching_set, batch_sampler=matching_sampler)
    """

    def __init__(
        self,
        dataset: TopicMatchingDataset,
        matching_size: int = 128,
        drop_last: bool = True,
        num_steps: int = None,  # step 수를 직접 정할 수 있음
        seed: int = None,
    ):
        if len(dataset.exhibit_ids) < matching_size:
            warnings.warn(
                f"'matching_size({matching_size})' is greater than the number of plans({len(dataset.exhibit_ids)}). 'matching_size' will be modified: ({matching_size}) -> ({len(dataset.exhibit_ids)})"
            )
            matching_size = len(dataset.exhibit_ids)
        super().__init__(data_source=dataset)

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
                    if len(self.dataset.exhibit_ids) >= self.last_batch_size
                    else True
                )
                groups = np.random.choice(
                    self.dataset.exhibit_ids, self.last_batch_size, replace=replace
                )
                if len(groups) != len(set(groups)):
                    raise ValueError("sampled groups are not unique for each other.")
            else:
                replace = (
                    False if len(self.dataset.exhibit_ids) >= self.batch_size else True
                )
                groups = np.random.choice(
                    self.dataset.exhibit_ids, self.batch_size, replace=replace
                )
                if len(groups) != len(set(groups)):
                    raise ValueError("sampled groups are not unique for each other.")
            batches.append(groups)
        return iter(batches)

    def __len__(self):
        return self.num_steps


class ImageRankingBatchSampler(Sampler):
    """Image Ranking Task를 위한 Batch Sampler
    Image Ranking Task: 기획전 상품 중 기준 지표(criterion)가 가장 높을 만 한 이미지를 추출하는 metric learning

    Args:
        dataset (ImageRankingDataset)
        sampling_iter: 매 step마다 샘플링할 횟수
          - one_step_one_plan=True: 하나의 기획전으로부터 sampling_iter번 만큼 샘플링을 수행
          - one_step_one_plan=False: sampling_iter개의 기획전으로부터 1번씩 샘플링을 수행
        replace:
          - False: 모든 iterating 동안 겹치는 기획전이 없도록 샘플링을 수행

    Examples:
        ranking_set = ImageRankingDataset(meta, img_dir, ...)
        ranking_sampler = ImageRankingBatchSampler(ranking_set)
        ranking_loader = Dataloader(ranking_set, batch_sampler=ranking_sampler)
    """

    def __init__(
        self,
        dataset: ImageRankingDataset,
        sampling_iter: int = 3,
        one_step_one_plan: bool = True,
        seed: int = None,
    ):
        if not one_step_one_plan and len(dataset.exhibit_ids) < sampling_iter:
            raise ValueError(
                f"There are {len(dataset.exhibit_ids)} exhibitions in meta data that is insufficient for sampling."
            )
        super().__init__(data_source=dataset)
        self.dataset = dataset
        self.sampling_iter = sampling_iter
        self.one_step_one_plan = one_step_one_plan
        self.replace = (
            False
            if not one_step_one_plan and len(dataset.exhibit_ids) >= sampling_iter
            else True
        )
        self.num_steps = (
            len(dataset.exhibit_ids)
            if self.replace or one_step_one_plan
            else len(dataset.exhibit_ids) // sampling_iter
        )
        self.seed = seed

    def __iter__(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.one_step_one_plan:
            batches = np.random.choice(
                self.dataset.exhibit_ids, self.num_steps, replace=False
            )[:, np.newaxis]
            batches = np.repeat(batches, self.sampling_iter, axis=1).tolist()

        else:
            batches = []
            if self.replace:
                for _ in range(self.num_steps):
                    batch = np.random.choice(
                        self.dataset.exhibit_ids, self.sampling_iter, replace=False
                    ).tolist()
                    batches.append(batch)
            else:
                shuffled_plan_ids = deepcopy(self.dataset.exhibit_ids)
                random.shuffle(shuffled_plan_ids)
                for _ in range(self.num_steps):
                    batch = [shuffled_plan_ids.pop() for _ in range(self.sampling_iter)]
                    batches.append(batch)

        return iter(batches)

    def __len__(self):
        return self.num_steps
