import os
import re
from time import time
from typing import Dict, Union

import numpy as np
import pandas as pd
from utils import load_pickle


def recover_url(x):
    """대체 url 코드 복원
    Reference. https://namu.wiki/w/URL%20escape%20code
    """
    x = x.replace("%20", " ")
    x = x.replace("%21", "!")
    x = x.replace("%22", '"')
    x = x.replace("%23", "#")
    x = x.replace("%24", "$")
    x = x.replace("%25", "%")
    x = x.replace("%26", "&")
    x = x.replace("%27", "'")
    x = x.replace("%28", "(")
    x = x.replace("%29", ")")
    x = x.replace("%2A", "*")
    x = x.replace("%2B", "+")
    x = x.replace("%2C", ",")
    x = x.replace("%2D", "-")
    x = x.replace("%2E", ".")
    x = x.replace("%2F", "/")
    x = x.replace("%3A", ":")
    x = x.replace("%3B", ";")
    x = x.replace("%3C", "<")
    x = x.replace("%3D", "=")
    x = x.replace("%3E", ">")
    x = x.replace("%3F", "?")
    x = x.replace("%40", "@")
    x = x.replace("%5B", "[")
    x = x.replace("%5C", "\\")
    x = x.replace("%5D", "]")
    x = x.replace("%5E", "^")
    x = x.replace("%5F", "_")
    x = x.replace("%60", "`")
    x = x.replace("%7B", "{")
    x = x.replace("%7C", "|")
    x = x.replace("%7D", "}")
    x = x.replace("%7E", "~")
    return x


def get_plan_id(x) -> int:
    format_1 = "https://m.shopping.naver.com/plan/details/"
    format_2 = "https://m.shopping.naver.com/plan2/m/preview.nhn?seq="
    format_3 = "trx="

    if "undefined" in x:
        plan_id = np.nan
    elif format_1 in x:
        p = re.compile(r"details/[0-9]+")
        plan_id = int(p.findall(x)[0].split("/")[-1])
    elif format_2 in x:
        p = re.compile(r"seq=[0-9]+")
        plan_id = int(p.findall(x)[0].split("=")[-1])

    elif format_3 in x:
        p = re.compile(r"trx=[0-9]+")
        plan_id = int(p.findall(x)[0].split("=")[-1])
    else:
        plan_id = np.nan

    return plan_id


def collect_click_logs(
    start_date: str,
    end_date: str,
    key: Union[str, Dict[str, str]] = None,
    save_dir: str = "../../data/plan_data/click_logs/",
    return_data: bool = False,
):
    """클릭 로그 수집 함수

    Args.
        key: 다음 형태의 key 정보
            {'location': 'LOCATION_INFO',
             'ticket': 'TICKET_ID',
             'principal': 'PRINCIPAL',
             'keytab_file': 'KEYTAB_PATH',
             'config': CONFIG}
    """
    try:
        from cuve import pylagos  # NOTE. 사용할 수 있는 환경이 제한적
    except:
        raise ImportError("There's no library 'cuve'")

    if isinstance(key, str):
        assert os.path.isfile(key), f"There's no file {key}"
        key = load_pickle(key)

    connect = pylagos.connect(**key)

    query_0 = f"""
    SELECT log_timestamp,gdid,lcookie,area,target_url
    FROM korea_logdata_stream.nclicks
    WHERE
        nsccode = 'Mshopping.plan' AND
        area IN ('plc.item','plb.item','plh.planname','tag.planname') AND
        lcookie != 'None' AND
        log_date BETWEEN \"{start_date}\" AND \"{end_date}\"
    """
    query_1 = f"""
    SELECT log_timestamp,gdid,lcookie,area,target_url
    FROM korea_logdata_stream.nclicks
    WHERE
        nsccode = 'plan.all' AND
        area = 'hom.name' AND
        lcookie != 'None' AND
        log_date BETWEEN \"{start_date}\" AND \"{end_date}\"
    """
    start = time()
    clicks_0 = pd.read_sql(query_0, connect)
    clicks_1 = pd.read_sql(query_1, connect)
    print(f"Time: {start - time() / 60:.2f} minutes")

    clicks = pd.concat([clicks_0, clicks_1], ignore_index=True)
    save_path = os.path.join(save_dir, f"clicks_{start_date}_{end_date}.csv")
    clicks.to_csv(save_path, index=False)
    print(f"Click logs have beed saved: '{save_path}'")

    if return_data:
        return clicks


def attach_ctr_label(meta: pd.DataFrame, click_logs: pd.DataFrame) -> pd.DataFrame:
    """기획전 내 각 상품에 대한 CTR 계산 함수.
    실행 시 기존 meta 데이터에 clicks, impression, ctr의 3가지 컬럼이 추가됨
        - clicks: 각 상품의 클릭 수
        - impressions: 각 상품의 노출 수
        - ctr: 각 상품의 클릭률
    """

    def _get_impressions(
        prod_id: int, prod_belong_plans: dict, views_per_plan: dict
    ) -> int:
        """impression 추출"""
        belong_plans = prod_belong_plans[prod_id]
        impressions = 0
        for p in belong_plans:
            impressions += views_per_plan[p]
        return impressions

    # 클릭이 발생한 기간으로 데이터 한정
    min_date = click_logs["log_timestamp"].min().split()[0]  # YY-MM-DD
    max_date = click_logs["log_timestamp"].max().split()[0]  # YY-MM-DD
    print(f"Date: {min_date} - {max_date}")

    meta = meta[
        meta["plan_startdate"].apply(lambda x: min_date <= x <= max_date)
    ].reset_index(drop=True)

    # 활용 가능한 기획전/상품만 필터링
    prod_ids = meta["prod_id"].unique().tolist()
    plan_ids = meta["plan_id"].unique().tolist()
    click_logs = click_logs[click_logs["gdid"].isin(prod_ids + plan_ids)].reset_index(
        drop=True
    )

    # 기획전 ID 추출
    click_logs.loc[:, "target_url"] = click_logs.loc[:, "target_url"].apply(
        lambda x: recover_url(x)
    )
    click_logs.loc[:, "plan_id"] = click_logs.loc[:, "target_url"].apply(
        lambda x: get_plan_id(x)
    )

    # 활용 가능한 기획전/상품만 필터링
    click_logs = click_logs[click_logs["plan_id"].notnull()].reset_index(drop=True)
    click_logs["plan_id"] = click_logs["plan_id"].astype(int)

    # 상품 ID 추출
    click_logs["prod_id"] = -1
    click_logs.loc[click_logs["gdid"].isin(prod_ids), "prod_id"] = click_logs.loc[
        click_logs["gdid"].isin(prod_ids), "gdid"
    ]

    plan_prod_clicks = (
        click_logs.groupby("plan_id")
        .apply(lambda x: x["prod_id"].value_counts())
        .reset_index()
    )
    plan_prod_clicks.columns = ["plan_id", "prod_id", "clicks"]

    # 각 기획전의 페이지 뷰 수
    views_per_plan = plan_prod_clicks.groupby("plan_id")["clicks"].sum().to_dict()

    # 각 상품의 클릭 수
    prod_belong_plans = dict(tuple(plan_prod_clicks.groupby("prod_id")["plan_id"]))
    prod_belong_plans = {
        prod_id: prod_belong_plans[prod_id].values.tolist()
        for prod_id in prod_belong_plans.keys()
    }

    # 각 상품의 클릭률
    prods_ctr = (
        plan_prod_clicks[plan_prod_clicks["prod_id"] != -1]
        .groupby("prod_id", as_index=False)["clicks"]
        .sum()
    )
    prods_ctr["impressions"] = prods_ctr["prod_id"].apply(
        lambda x: _get_impressions(x, prod_belong_plans, views_per_plan)
    )
    prods_ctr["ctr"] = prods_ctr["clicks"] / prods_ctr["impressions"]
    assert len(prods_ctr) == prods_ctr["prod_id"].nunique()

    # CTR 관련 컬럼 추가
    meta_ctr_labeled = meta.merge(prods_ctr, how="left", on="prod_id")
    meta_ctr_labeled.loc[
        meta_ctr_labeled["impressions"].isnull(), "impressions"
    ] = meta_ctr_labeled.loc[meta_ctr_labeled["impressions"].isnull(), "plan_id"].apply(
        lambda x: views_per_plan.get(x, 0)
    )
    meta_ctr_labeled[["clicks", "ctr"]] = meta_ctr_labeled[["clicks", "ctr"]].fillna(0)
    meta_ctr_labeled[["clicks", "impressions"]] = meta_ctr_labeled[
        ["clicks", "impressions"]
    ].astype(int)

    return meta_ctr_labeled


if __name__ == "__main__":
    pass
