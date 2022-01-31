from typing import Union, Tuple
import numpy as np
import pandas as pd
from utils.data_utils import train_test_split_groupby_plan, resolve_data_leakage

DEFAULT_REFINE_RULE = {
    "target": "ctr",
    "sample_size": 50,  # None:
    "unique_constraint": 2,  # 기획전 내 target의 uniqueness가 unique_constraint보다 큰 기획전만 필터링
    "min_nonzero_num": 5,  # 각 기획전의 리뷰수/판매수/찜수 각각이 0이 아닌 상품이 최소 몇 개 있어야 하는지에 대한 조건
}


class PlanDataRefiner:
    """PlanDataCollector를 통해 수집된 raw 데이터를 학습에 활용 가능한 형태로 재구성하는 클래스
    각 기획전으로부터 균형 있게 상품을 샘플링 하는 등의 과정을 거침

    Refinement Process
        1. filter_nonzero_num_over_k
            - 리뷰수가 0개 보다 많은 상품이 k개 이상인 기획전만을 추출
            - target에 따른 상대적 비교, 즉 target에 따른 positive/negative를 원활히 매기기 위함
        2. subsample_each_plan
            - 각 기획전으로부터 sample_size만큼 상품 샘플을 추출
            - 각 기획전의 모델 학습에 대한 contribution을 맞추기 위함
        3. remove_sparse_plans
            - 각 기획전의 리뷰수/구매수/찜수에 대한 uniqueness 수가 unique_constraint 이하인 케이스를 제외
            - 기준값에 따른 상대적 비교, 즉 기준값에 따른 positive/negative를 원활히 매기기 위함

    Example:
        raw = pd.read_csv("raw.csv")
        refiner = PlanDataRefiner()
        train_refined = refiner.sift(raw)
    """

    NECESSARY_RULE_KEYS = [
        "target",
        "sample_size",
        "unique_constraint",
        "min_nonzero_num",
    ]

    def __init__(self, rule: dict, seed: int = 27):
        self.target = rule["target"]
        self.rule = rule if rule is not None else DEFAULT_REFINE_RULE
        self.seed = seed

        # verify refinement rules
        for k in self.NECESSARY_RULE_KEYS:
            assert k in self.rule, f"'{k}' arg not in rule instance"

    def sift(
        self,
        raw: pd.DataFrame,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """raw 데이터 정제 함수

        Args:
            raw (pd.DataFrame): PlanDataCollector를 통해 수집한 raw 데이터
            verbose (bool, optional): [description]. Defaults to False.

        Returns:
            pd.DataFrame: [description]
        """
        print(f"[+] Refinement Progress - # Raw Data: {len(raw):,d}")
        if verbose:
            self.verbose_category_desc(raw)

        train_filtered = self.filter_nonzero_num_over_k(
            raw, self.rule["min_nonzero_num"]
        )

        print(
            f">> {self.target}이(가) 0보다 큰 상품이 {self.rule['min_nonzero_num']}개 이상인 기획전만 추출: {len(train_filtered):,d}"
        )
        if verbose:
            self.verbose_category_desc(train_filtered)

        if self.rule["sample_size"] is not None:
            train_filtered = self.subsample_each_plan(
                train_filtered, self.rule["sample_size"]
            )
            print(
                f">> 각 기획전으로부터 {self.rule['sample_size']}개 상품을 샘플링(weight: {self.target}): {len(train_filtered):,d}"
            )
            if verbose:
                self.verbose_category_desc(train_filtered)

        train_filtered = self.remove_sparse_plans(
            train_filtered, self.rule["unique_constraint"]
        )
        print(
            f">> '{self.target}'에 대한 uniqueness가 {self.rule['unique_constraint']} 이상인 기획전만 추출: {len(train_filtered):,d}"
        )
        if verbose:
            self.verbose_category_desc(train_filtered)

        print("[+] Refinement Result\n", "")

        return train_filtered

    def filter_nonzero_num_over_k(self, raw, k: int) -> pd.DataFrame:
        nonzero_cnts = raw.groupby("plan_id")[self.target].apply(
            lambda x: (x > 0).sum() >= k
        )
        rich_plan_ids = nonzero_cnts[nonzero_cnts == True].index.tolist()
        raw = raw[raw["plan_id"].isin(rich_plan_ids)]
        raw = raw.reset_index(drop=True)
        return raw

    def subsample_each_plan(self, raw: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        np.random.seed(self.seed)
        rich_plan_ids = raw["plan_id"].value_counts()
        rich_plan_ids = rich_plan_ids[rich_plan_ids >= sample_size].index.tolist()
        raw = raw[raw["plan_id"].isin(rich_plan_ids)]

        raw["weight"] = (
            raw[self.target].values + np.random.uniform(size=len(raw)) * 1e-8
        )
        raw = raw.groupby("plan_id").apply(
            lambda x: x.sample(sample_size, weights="weight", replace=False)
        )
        raw = raw.reset_index(drop=True)
        raw = raw.drop("weight", axis=1)
        return raw

    def remove_sparse_plans(
        self, raw: pd.DataFrame, unique_constraint: int = 2
    ) -> pd.DataFrame:
        """기획전의 target value의 uniqueness가 unique_constraint보다 작을 경우 버림"""
        target_nunq = raw.groupby("plan_id")[self.target].nunique()
        poor_plan_ids = target_nunq[target_nunq < unique_constraint].index.tolist()
        poor_plan_ids = list(set(poor_plan_ids))
        raw = raw[~raw["plan_id"].isin(poor_plan_ids)]
        raw = raw.reset_index(drop=True)
        return raw

    @staticmethod
    def verbose_category_desc(meta: pd.DataFrame):
        plan_cat1_cnts = meta["plan_cat1"].value_counts()
        plan_cat1_cnts = dict(zip(plan_cat1_cnts.index, plan_cat1_cnts.values))
        plan_cat2_cnts = meta["plan_cat2"].value_counts()
        plan_cat2_cnts = dict(zip(plan_cat2_cnts.index, plan_cat2_cnts.values))
        prod_cat1_cnts = meta["prod_cat1"].value_counts()
        prod_cat1_cnts = dict(zip(prod_cat1_cnts.index, prod_cat1_cnts.values))
        prod_cat2_cnts = meta["prod_cat2"].value_counts()
        prod_cat2_cnts = dict(zip(prod_cat2_cnts.index, prod_cat2_cnts.values))

        descriptions = dict(
            plan_cat1=plan_cat1_cnts,
            plan_cat2=plan_cat2_cnts,
            prod_cat1=prod_cat1_cnts,
            prod_cat2=prod_cat2_cnts,
        )
        print("Prod Count for Each Category:")
        for key, cnt_dict in descriptions.items():
            print(f"  <{key}>")
            for cat, cnt in cnt_dict.items():
                print(f"   * {cat}: {cnt:,d}")
            print()


def train_valid_test_split(
    meta: pd.DataFrame,
    train_size: float = 0.7,
    valid_size: float = 0.2,
    test_size: float = 0.1,
    eval_cutoff: bool = True,
    target: str = "ctr",
    deleakage: bool = True,
    random_state: int = 27,
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]
]:
    """학습/검증/테스트 데이터 split 함수. split 시 기획전 카테고리(depth=2)를 기준으로 stratified sampling 됨

    NOTE. test_size를 0으로 설정 시 학습/검증 데이터만 리턴 됨

    Args:
        meta (pd.DataFrame): 학습용 메타데이터
        train_size (float, optional): 학습 데이터 비율. Defaults to 0.7.
        valid_size (float, optional): 검증 데이터 비율. Defaults to 0.2.
        test_size (float, optional): 테스트 데이터 비율. Defaults to 0.1.
        target (str, optional): 타깃컬럼명. Defaults to "ctr".
        deleakage (bool, optional): data leakage 제거 여부. Defaults to True.
        random_state (int, optional): 시드값. Defaults to 27.
    """
    # rescale
    train_size = train_size / (train_size + valid_size + test_size)
    valid_size = valid_size / (train_size + valid_size + test_size)
    test_size = test_size / (train_size + valid_size + test_size)

    print(
        f"[+] Data Split\n",
        f"Train Size: {train_size:.2f}\n",
        f"Valid Size: {valid_size:.2f}\n",
        f"Test Size: {test_size:.2f}\n",
        f"De-Leakage: {deleakage}\n",
        f"Seed: {random_state}\n",
    )
    train, valid = train_test_split_groupby_plan(
        meta, train_size, random_state=random_state
    )

    if test_size > 0.0:
        valid_size_rescaled = valid_size / (valid_size + test_size)
        valid, test = train_test_split_groupby_plan(
            valid, valid_size_rescaled, random_state=random_state
        )
        if eval_cutoff:
            test = subsample_each_plan(
                test, sample_size=50, target=target, seed=random_state
            )

    if eval_cutoff:
        valid = subsample_each_plan(
            valid, sample_size=50, target=target, seed=random_state
        )

    if deleakage:
        if test_size > 0.0:
            train, valid = resolve_data_leakage(train, valid, target=target)
            train, test = resolve_data_leakage(train, test, target=target)
        else:
            train, valid = resolve_data_leakage(train, valid, target=target)

    if test_size > 0.0:
        print(
            "[+] Split Result\n",
            f"# Plans for Train: {train['plan_id'].nunique():,d}\n",
            f"# Plans for Valid: {valid['plan_id'].nunique():,d}\n",
            f"# Plans for Test: {test['plan_id'].nunique():,d}\n",
            f"# Products for Train: {train['prod_id'].nunique():,d}\n",
            f"# Products for Valid: {valid['prod_id'].nunique():,d}\n",
            f"# Products for Test: {test['prod_id'].nunique():,d}\n",
        )
        return train, valid, test

    else:
        print(
            "[+] Split Result\n",
            f"# Plans for Train: {train['plan_id'].nunique():,d}\n",
            f"# Plans for Valid: {valid['plan_id'].nunique():,d}\n",
            f"# Products for Train: {train['prod_id'].nunique():,d}\n",
            f"# Products for Valid: {valid['prod_id'].nunique():,d}\n",
        )
        return train, valid


def subsample_each_plan(
    raw: pd.DataFrame, sample_size: int, target: str = "ctr", seed: int = 27
) -> pd.DataFrame:
    np.random.seed(seed)
    rich_plan_ids = raw["plan_id"].value_counts()
    rich_plan_ids = rich_plan_ids[rich_plan_ids >= sample_size].index.tolist()
    raw = raw[raw["plan_id"].isin(rich_plan_ids)]

    raw["weight"] = raw[target].values + np.random.uniform(size=len(raw)) * 1e-8
    raw = raw.groupby("plan_id").apply(
        lambda x: x.sample(sample_size, weights="weight", replace=False)
    )
    raw = raw.reset_index(drop=True)
    raw = raw.drop("weight", axis=1)
    return raw
