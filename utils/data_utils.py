import warnings
from typing import Tuple, List, Union, Dict, Optional

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    TopicMatchingDataset,
    ImageRankingDataset,
    TopicMatchingBatchSampler,
    ImageRankingBatchSampler,
)
from data_collection.data_collection import PLAN_CATS_EN2KR
from preprocessing import TextPreprocessor
from .ddp_utils import DistributedBatchSampler
from .flags import Flags


def compose_dataloaders(
    meta_matching: pd.DataFrame,
    meta_discrim: pd.DataFrame,
    labeling_criterion: str,
    img_dir: str,
    img_transforms: A.Compose,
    txt_preprocessor: TextPreprocessor,
    exhibit_attrs: List[str],
    prod_attrs: List[str] = None,
    matching_size: int = 512,
    ranking_size: int = 20,
    ranking_iter: int = 12,
    sampling_method: str = "weighted",
    txt_aug_prob: float = 0.0,
    num_workers: int = 1,
    rank: int = None,
) -> Tuple[DataLoader, DataLoader]:
    is_distributed = True if rank is not None else False

    matching_set = TopicMatchingDataset(
        meta=meta_matching,
        labeling_criterion=labeling_criterion,
        img_dir=img_dir,
        img_transforms=img_transforms,
        txt_preprocessor=txt_preprocessor,
        exhibit_attrs=exhibit_attrs,
        prod_attrs=prod_attrs,
        txt_aug_prob=txt_aug_prob,
    )
    ranking_set = ImageRankingDataset(
        meta=meta_discrim,
        labeling_criterion=labeling_criterion,
        img_dir=img_dir,
        img_transforms=img_transforms,
        txt_preprocessor=txt_preprocessor,
        exhibit_attrs=exhibit_attrs,
        group_sampling_size=ranking_size,
        sampling_method=sampling_method,
    )

    ranking_sampler = ImageRankingBatchSampler(ranking_set, ranking_iter)

    # 두 데이터셋의 배치 사이즈 동기화
    if is_distributed and len(ranking_sampler) < matching_size:
        matching_sampler = TopicMatchingBatchSampler(
            dataset=matching_set,
            matching_size=(len(ranking_sampler) // dist.get_world_size())
            * dist.get_world_size(),
            num_steps=len(ranking_sampler),
        )
        warnings.warn(
            f"'matching_size' should be multiple of world_size({dist.get_world_size()}) during DDP. 'matching_size' will be modified: {matching_size} -> {(len(ranking_sampler) // dist.get_world_size()) * dist.get_world_size()}"
        )
    else:
        matching_sampler = TopicMatchingBatchSampler(
            matching_set, matching_size, num_steps=len(ranking_sampler)
        )

    if is_distributed:
        ranking_sampler = DistributedBatchSampler(
            ranking_sampler, num_replicas=dist.get_world_size(), rank=rank
        )
        matching_sampler = DistributedBatchSampler(
            matching_sampler, num_replicas=dist.get_world_size(), rank=rank
        )
        matching_loader = DataLoader(
            matching_set, batch_sampler=matching_sampler, num_workers=num_workers
        )
        ranking_loader = DataLoader(
            ranking_set, batch_sampler=ranking_sampler, num_workers=num_workers
        )
    else:
        matching_loader = DataLoader(
            matching_set, batch_sampler=matching_sampler, num_workers=num_workers
        )
        ranking_loader = DataLoader(
            ranking_set, batch_sampler=ranking_sampler, num_workers=num_workers
        )
    return matching_loader, ranking_loader


def train_test_split_group_by_plan(
    meta: pd.DataFrame,
    train_size: float = 0.8,
    stratify: str = "exhibit_cat2",
    random_state: int = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if train_size >= 1:
        return meta, None

    if random_state is not None:
        np.random.seed(random_state)

    # stratified split
    tmp_stratify = meta[meta[stratify].notnull()][["exhibit_id", stratify]]
    tmp_stratify = tmp_stratify.drop_duplicates(
        subset=["exhibit_id"]
    )  # make unique to group exhibit_id

    class_cnt = tmp_stratify[stratify].value_counts()
    filtered_classes = class_cnt[
        class_cnt > 1
    ].index.tolist()  # remove class whose count is 1

    tmp_stratify = tmp_stratify[tmp_stratify[stratify].isin(filtered_classes)]
    train_plans, test_plans = train_test_split(
        tmp_stratify,
        train_size=train_size,
        stratify=tmp_stratify[stratify],
        shuffle=True,
        random_state=random_state,
    )
    train_exhibitions = train_plans["exhibit_id"].tolist()
    test_exhibitions = test_plans["exhibit_id"].tolist()
    train_meta = meta[meta["exhibit_id"].isin(train_exhibitions)].reset_index(drop=True)
    test_meta = meta[meta["exhibit_id"].isin(test_exhibitions)].reset_index(drop=True)
    return train_meta, test_meta


def load_meta_data(
    args: Flags,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    partial_cats = [PLAN_CATS_EN2KR[c] for c in args.partial_cats]

    def _filter(m: pd.DataFrame, d: pd.DataFrame):
        if partial_cats:
            m = m[m[args.main_cat_depth].isin(partial_cats)].reset_index(drop=True)
            d = d[d[args.main_cat_depth].isin(partial_cats)].reset_index(drop=True)
        plan_freqs = d["plan_id"].value_counts()
        useful_plan_ids = plan_freqs[plan_freqs >= 50].index.tolist()
        d = d[d["plan_id"].isin(useful_plan_ids)].reset_index(drop=True)
        return m, d

    def _split_matching_ranking(
        m: pd.DataFrame, d: pd.DataFrame, subsample_rate: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        d, _ = train_test_split_group_by_plan(d, subsample_rate)

        # matching will be made with considering discrim
        tmp_m_0 = m[m["exhibit_id"].isin(d["exhibit_id"].unique())]
        if (
            subsample_rate
            - (tmp_m_0["exhibit_id"].nunique() / m["exhibit_id"].nunique())
            > 0
        ):
            try:
                tmp_m_1, _ = train_test_split_group_by_plan(
                    m[~m["exhibit_id"].isin(tmp_m_0["exhibit_id"].unique())],
                    train_size=subsample_rate
                    - (tmp_m_0["exhibit_id"].nunique() / m["exhibit_id"].nunique()),
                )
                m = pd.concat([tmp_m_0, tmp_m_1], axis=0, ignore_index=True)
            except:  # groupby split을 하기에는 남은 데이터셋이 너무 작을 경우 => tmp_m_0 무시하고 새로 만듦
                m, _ = train_test_split_group_by_plan(m, train_size=subsample_rate)
        elif (
            subsample_rate
            - (tmp_m_0["exhibit_id"].nunique() / m["exhibit_id"].nunique())
            == 0
        ):
            m = tmp_m_0.reset_index(drop=True)
        else:
            raise NotImplementedError

        return m, d

    # train
    train_m, train_d = pd.read_csv(args.train_matching), pd.read_csv(args.train_ranking)
    train_m, train_d = _filter(train_m, train_d)
    train_m, train_d = _split_matching_ranking(
        train_m, train_d, args.train_subsample_rate
    )

    # valid
    valid_m, valid_d = pd.read_csv(args.valid_matching), pd.read_csv(args.valid_ranking)
    valid_m, valid_d = _filter(valid_m, valid_d)
    valid_m, valid_d = _split_matching_ranking(
        valid_m, valid_d, args.valid_subsample_rate
    )

    train_m, valid_m = resolve_data_leakage(
        train_m, valid_m, labeling_criterion=args.labeling_criterion
    )
    train_d, valid_d = resolve_data_leakage(
        train_d, valid_d, labeling_criterion=args.labeling_criterion
    )
    return train_m, train_d, valid_m, valid_d


def resolve_data_leakage(
    train: pd.DataFrame, valid: pd.DataFrame, labeling_criterion: str = "ctr"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """valid에 포함된 상품 이미지를 train에 등장하지 않도록 전처리
    vplan_ver_2-2 정제 결과 기획전 수:
    - Train: 5406 -> 1262
    - Valid: 576 -> 343
    """
    valid_plan_names = valid["exhibit_name"].unique().tolist()
    valid_prod_ids = valid["prod_id"].unique().tolist()

    # valid에 등장한 상품 모두 제거
    # 주의: 컨셉이 아예 같은 기획전인 경우 train과 valid 분포가 틀어질 수 있음
    train_filtered = train[
        (~train["prod_id"].isin(valid_prod_ids))
        & (~train["exhibit_name"].isin(valid_plan_names))
    ].reset_index(drop=True)

    # train 내 필터링된 상품이 50개 미만인 기획전 제거
    train_exhibit_freqs = train_filtered["exhibit_id"].value_counts()
    train_useful_exhibit_ids = train_exhibit_freqs[
        train_exhibit_freqs >= 50
    ].index.tolist()
    train_filtered = train_filtered[
        train_filtered["exhibit_id"].isin(train_useful_exhibit_ids)
    ].reset_index(drop=True)

    # 학습에 쓸 만한 기획전만
    num_nonzeros_by_exhibition = train_filtered.groupby("exhibit_id").apply(
        lambda x: (x[labeling_criterion] != 0).sum()
    )
    label_nunique = train_filtered.groupby("exhibit_id").apply(
        lambda x: x[labeling_criterion].nunique()
    )
    train_useful_exhibit_ids = list(
        set(
            num_nonzeros_by_exhibition[num_nonzeros_by_exhibition < 5].index.tolist()
            + label_nunique[label_nunique > 1].index.tolist()
        )
    )
    train_filtered = train_filtered[
        train_filtered["exhibit_id"].isin(train_useful_exhibit_ids)
    ].reset_index(drop=True)
    train_filtered = train_filtered.sort_values(
        by=["exhibit_id", labeling_criterion], ascending=False, ignore_index=True
    )

    # de-duplicate: 기획전 간 중복 상품 제거
    train_filtered = make_meta_unique(
        train_filtered, labeling_criterion=labeling_criterion
    )
    valid_filtered = make_meta_unique(valid, labeling_criterion=labeling_criterion)
    return train_filtered, valid_filtered


def make_meta_unique(
    meta: pd.DataFrame, labeling_criterion: str = "ctr"
) -> pd.DataFrame:
    # train 기획전 간 uniqueness 부여
    cumul_prod_ids = set()
    unique_exhibitions = []
    exhibit_ids = meta["exhibit_id"].unique().tolist()

    # de-duplicate
    tmp = meta[["exhibit_id", "exhibit_name"]].drop_duplicates(ignore_index=True)
    dedup_locs = tmp["exhibit_name"].drop_duplicates().index.tolist()
    dedup_plan_ids = tmp.loc[dedup_locs]["exhibit_id"].tolist()
    dedup = meta[meta["exhibit_id"].isin(dedup_plan_ids)].reset_index(drop=True)

    for p in tqdm(exhibit_ids):
        tmp = dedup[
            (dedup["exhibit_id"] == p) & (~dedup["prod_id"].isin(cumul_prod_ids))
        ]
        if (
            len(tmp) < 50
            or tmp[labeling_criterion].nunique() < 2
            or (tmp[labeling_criterion] == 0).sum() < 5
        ):
            continue
        if len(tmp) > 100:
            tmp = tmp.head(75)
        unique_exhibitions.append(tmp)
        cumul_prod_ids.update(tmp["prod_id"].unique().tolist())

    meta_unique = pd.concat(unique_exhibitions, ignore_index=True)
    assert len(meta_unique["prod_id"]) == meta_unique["prod_id"].nunique()
    return meta_unique


def compose_batch(
    topics: Dict[str, torch.Tensor], images: torch.Tensor, device: torch.device
) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
    batch = {"topics": txt_input_to_device(topics, device), "images": images.to(device)}
    return batch


def txt_input_to_device(
    txt_input: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    txt_input["input_ids"] = txt_input["input_ids"].to(device)
    txt_input["token_type_ids"] = txt_input["token_type_ids"].to(device)
    txt_input["attention_mask"] = txt_input["attention_mask"].to(device)
    return txt_input
