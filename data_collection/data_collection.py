import argparse
import os
import json
import warnings
from time import time
from collections import defaultdict
from typing import Dict
import requests
from tqdm import tqdm
from datetime import datetime, timedelta
import cv2
import pandas as pd
import numpy as np
from utils import load_pickle, save_pickle

PROD_REQ_URL = "http://ss-api.flova-pipeline.svc.ad1.io.navercorp.com/v1.1/vertical/products/_lookup"
PROD_REQ_HEADERS = {"accept": "application/json", "Content-Type": "application/json"}
PROD_DROP_COLS = ["naverShoppingRegistrationYn", "authenticationType"]
PLAN_DROP_COLS = ["inspectionStatus", "serviceMappingId", "sections"]
PLAN_DATE_COLS = ["exposureStartAt", "exposureEndAt", "inspectedAt"]
PLAN_CATS_KR2EN = {
    "디지털": "digital",
    "레저/스포츠": "leisure_sports",
    "남성의류": "male_clothes",
    "남성신발/가방": "male_shoes_bags_accessories",
    "가구": "furniture",
    "침구/페브릭": "bed_fabric",
    "생활용품": "living_goods",
    "주방용품": "kitchen_goods",
    "인테리어": "interior",
    "여성의류": "women_clothes",
    "액세서리": "accessories",
    "언더웨어": "underware",
    "패션종합": "fashion_total",
    "여성슈즈": "women_shoes",
    "여성가방": "women_bags",
    "여성잡화": "women_goods",
    "유아동패션": "kids_fashion",
    "식품/건강": "food_health",
}
PLAN_CATS_EN2KR = {
    "digital": "디지털",
    "leisure_sports": "레저/스포츠",
    "male_clothes": "남성의류",
    "male_shoes_bags_accessories": "남성신발/가방",
    "furniture": "가구",
    "bed_fabric": "침구/페브릭",
    "living_goods": "생활용품",
    "kitchen_goods": "주방용품",
    "interior": "인테리어",
    "women_clothes": "여성의류",
    "accessories": "액세서리",
    "underware": "언더웨어",
    "fashion_total": "패션종합",
    "women_shoes": "여성슈즈",
    "women_bags": "여성가방",
    "women_goods": "여성잡화",
    "kids_fashion": "유아동패션",
    "food_health": "식품/건강",
}
PROD_COLS = [
    "prod_id",
    "category1Name",
    "category2Name",
    "category3Name",
    "category4Name",
    "brand",
    "reviewCount",
    "productTitle",
    "title",
    "bodyText",
    "zzimCount",
    "saleCount",
    "openDate",
]
PLAN_COLS = [
    "plan_id",
    "name",
    "category1Name",
    "category2Name",
    "keyword",
    "exposureStartAt",
    "exposureEndAt",
    "templateTitle",
    "templateText",
]
PROD_RENAME_DICT = {
    "category1Name": "prod_cat1",
    "category2Name": "prod_cat2",
    "category3Name": "prod_cat3",
    "category4Name": "prod_cat4",
    "brand": "prod_brand",
    "reviewCount": "prod_review_cnt",
    "productTitle": "prod_name",
    "title": "prod_page_title",
    "bodyText": "prod_text",
    "zzimCount": "prod_zzim_cnt",
    "saleCount": "prod_sale_cnt",
    "openDate": "prod_opendate",
}
PLAN_RENAME_DICT = {
    "name": "plan_name",
    "category1Name": "plan_cat1",
    "category2Name": "plan_cat2",
    "keyword": "plan_kwds",
    "exposureStartAt": "plan_startdate",
    "exposureEndAt": "plan_enddate",
    "templateTitle": "plan_page_title",
    "templateText": "plan_text",
}

KEY_PATH = "/home/prev_data/aimd/jeehyungko/data/plan_data/private_key.pkl"


class PlanDataCollector:
    """버티컬 기획전 데이터 수집 클래스

    Example - Notebook
        start_date = '2020-09-05'
        end_date = '2021-09-05'
        save_dir = '../data/plan_data/'

        collector = PlanDataCollector()
        collector.collect(start_date, end_date, save_dir=save_dir)

    Example - Command Line Interface
        python data_collection.py --start_date '2020-09-05' --end_date '2021-09-05' --save_dir '../data/plan_data'
    """

    def __init__(
        self,
        plan_cols: list = None,
        plan_drop_cols: list = None,
        plan_date_cols: list = None,
        plan_cats_kr2en: dict = None,
        prod_cols: list = None,
        prod_info_url: str = None,
        prod_headers: dict = None,
        prod_drop_cols=None,
    ):
        self.plan_cols = plan_cols if plan_cols is not None else PLAN_COLS
        self.plan_drop_cols = (
            plan_drop_cols if plan_drop_cols is not None else PLAN_DROP_COLS
        )
        self.plan_date_cols = (
            plan_date_cols if plan_date_cols is not None else PLAN_DATE_COLS
        )
        self.plan_cats_kr2en = (
            plan_cats_kr2en if plan_cats_kr2en is not None else PLAN_CATS_KR2EN
        )
        self.prod_cols = prod_cols if prod_cols is not None else PROD_COLS
        self.prod_info_url = (
            prod_info_url if prod_info_url is not None else PROD_REQ_URL
        )
        self.prod_headers = (
            prod_headers if prod_headers is not None else PROD_REQ_HEADERS
        )
        self.prod_drop_cols = (
            prod_drop_cols if prod_drop_cols is not None else PROD_DROP_COLS
        )

    def collect(
        self,
        start_date: str,
        end_date: str,
        save_dir: str,
        save_only_first_image: bool = True,
        pre_resize: bool = True,
        return_data: bool = False,
    ):
        end_year, end_month, end_day = self.get_year_month_day(end_date)
        end_date = f"{end_year}-{end_month}-{end_day}" if end_date is None else end_date
        self.make_dirs(save_dir, start_date, end_date)
        print(
            "[+] Collect Vertical Plan Data\n",
            f"Start date: {start_date}\n",
            f"End date: {end_date}\n",
            f"Save directory: {save_dir}\n",
            f"Save only first image(attnId=1) for each product: {save_only_first_image}\n",
            f"Pre-resize Image to 512x512: {pre_resize}\n",
        )

        # 기획전 메타 데이터 수집
        plans_table = self.collect_plan_data(start_date, end_date)

        # 상품 메타 데이터 수집
        prods_table = self.collect_prod_data(plans_table)

        # 상품 이미지 수집
        self.collect_prod_imgs(
            prods_table, save_only_first_image, pre_resize=pre_resize
        )

        # raw 데이터 구성
        if return_data:
            raw = self.compose_raw_data(return_data=return_data)
            return raw
        self.compose_raw_data(return_data=return_data)

    def collect_plan_data(self, start_date, end_date) -> pd.DataFrame:
        """기획전 메타 정보 저장 함수"""
        print("[+] Collect Plan Information", end="\n ")
        tt = time()

        start_year, start_month, start_day = self.get_year_month_day(start_date)
        end_year, end_month, end_day = self.get_year_month_day(end_date)
        plans = self.read_plans()
        plans_table = pd.DataFrame(plans.values())

        # 컬렴명 변경 (id -> plan_id)
        plans_table = plans_table.rename({"id": "plan_id"}, axis=1)

        for c in self.plan_date_cols:
            plans_table.loc[plans_table[c].notnull(), c] = plans_table.loc[
                plans_table[c].notnull(), c
            ].apply(lambda x: self.extract_date(x))
            plans_table.loc[:, c] = pd.to_datetime(plans_table.loc[:, c])

        # 설정 기간에 맞게 데이터 필터링
        plans_table = plans_table[
            (
                plans_table["exposureStartAt"]
                >= pd.to_datetime(f"{start_year}-{start_month:0>2d}-{start_day:0>2d}")
            )
            & (
                plans_table["exposureStartAt"]
                < pd.to_datetime(f"{end_year}-{end_month:0>2d}-{end_day:0>2d}")
                + timedelta(1)
            )
        ]

        # 섹션 정보(=부가 정보)에 접근할 수 없는 기획전 제거
        plans_table = plans_table[plans_table["sections"].apply(lambda x: len(x) > 0)]

        # 각 기획전 메타 정보 저장
        for _, row in tqdm(
            plans_table[["plan_id", "sections"]].iterrows(), "[Save Plan Section Info]"
        ):
            plan_id, sections = row
            save_pickle(
                os.path.join(self.plan_section_meta_dir, f"{plan_id}.pkl"), sections
            )

        # 데이터 품질을 위해 검수 완료(CMPLINSP)된 기획전만 필터링
        plans_table = plans_table[plans_table["inspectionStatus"] == "CMPLINSP"]

        # 키워드, 상세 제목이 있는 기획전만 필터링
        empty = ""
        plans_table = plans_table[
            (plans_table["keyword"] != empty)
            & (plans_table["keyword"].notnull())
            & (plans_table["templateTitle"] != empty)
            & (plans_table["templateTitle"].notnull())
        ]

        # 카테고리(depth=2)가 있는 기획전만 필터링
        plans_table = plans_table[
            plans_table["category2Name"].apply(lambda x: x in self.plan_cats_kr2en)
        ]

        # 불필요한 컬럼 제거
        plans_table = plans_table.drop(self.plan_drop_cols, axis=1)

        # 게시일 기준 내림차순 정렬
        plans_table = plans_table.sort_values(
            by=["exposureStartAt", "exposureEndAt"], ascending=False, ignore_index=True
        )

        # 메타 정보 저장
        if self.plan_save_dir is not None:
            plans_table.to_csv(
                os.path.join(self.plan_save_dir, "plan_data_all.csv"), index=False
            )

        interval = time() - tt
        print(f"Time: {interval:.2f} seconds\n", f"Saved in: '{self.plan_save_dir}'\n")
        return plans_table

    def collect_prod_data(self, plans_table: pd.DataFrame) -> pd.DataFrame:
        """상품 메타 정보 저장 함수"""
        print("[+] Collect Product Information", end="\n ")
        tt = time()
        results = []

        # 각 기획전 내 상품 정보 수집
        prod_section_names = defaultdict(set)
        for _, plan in tqdm(plans_table.iterrows(), desc="[Collect Product Data]"):
            category = self.plan_cats_kr2en.get(plan["category2Name"], "unknown")
            product_ids = set()  # keep uniqueness

            plan_id = plan["plan_id"]
            sections = load_pickle(
                os.path.join(self.plan_section_meta_dir, f"{plan_id}.pkl")
            )
            for sec in sections:
                sec_name = sec["name"]
                tmp_prod_ids = [str(p) for p in sec["productIds"]]  # string for request
                product_ids.update(tmp_prod_ids)

                # section names for each product
                for p in tmp_prod_ids:
                    prod_section_names[int(p)].add(sec_name)

            product_ids = list(product_ids)  # unique product IDs

            # {'category': <Category Name>, 'productIDs': [<'Product ID'>]}
            data = json.dumps(dict(category=category, productIDs=product_ids))
            try:
                result = requests.post(
                    self.prod_info_url, headers=self.prod_headers, data=data
                ).json()["result"]
                result = [
                    r
                    for r in result
                    if r is not None and "images" in r and len(r["images"]) > 0
                ]
                if len(result) == 0:
                    continue

                # 각 상품 정보 저장
                for r in result:
                    prod_id = r["id"]

                    imgs = r.pop("images")
                    imgs_recomposed = dict()
                    for img_info in imgs:
                        imgs_recomposed[img_info["attId"]] = dict(
                            imageUrl=img_info["imageUrl"], adult=img_info["adult"]
                        )

                    # 상품별 이미지 url 저장
                    save_pickle(
                        os.path.join(self.prod_img_meta_dir, f"{prod_id}.pkl"),
                        imgs_recomposed,
                    )

                    # 상품별 이미지 OCR 정보 저장
                    if "ocr" in r:
                        ocr = r.pop("ocr")
                        save_pickle(
                            os.path.join(self.prod_ocr_meta_dir, f"{prod_id}.pkl"), ocr
                        )
                results.extend(result)

            except:  # 접근할 데이터가 없는 경우
                continue

        # 상품 메타 정보 저장
        prods_table = pd.DataFrame(results).rename({"id": "prod_id"}, axis=1)
        prods_table = self.refine_prods_meta(prods_table)
        prods_table["prod_id"] = prods_table["prod_id"].astype(int)
        prods_table["openDate"] = prods_table["openDate"].apply(
            lambda x: self.digits2datetime(x)
        )
        prods_table.to_csv(
            os.path.join(self.prod_save_dir, "prod_data_all.csv"), index=False
        )

        for key, values in prod_section_names.items():
            prod_section_names[key] = list(values)

        # 상품 섹션명 정보 저장
        save_pickle(
            os.path.join(self.prod_save_dir, "prod_section_names.pkl"),
            dict(prod_section_names),
        )
        interval = time() - tt
        print(f"Time: {interval:.2f} seconds\n", f"Saved in: {self.prod_save_dir}\n")
        return prods_table

    def collect_prod_imgs(
        self,
        prods_table: pd.DataFrame,
        save_only_first_image: bool,
        pre_resize: bool = True,
    ):
        """상품 이미지 수집 함수

        Args:
            prods_table (pd.DataFrame): 상품 메타 정보 데이터프레임
            save_only_first_image (bool): True일 경우 대표 이미지(attnId=1)만을 저장
        """
        print("[+] Collect Product Images", end="\n ")
        tt = time()
        prod_ids = prods_table["prod_id"].tolist()  # unique list
        for prod_id in tqdm(prod_ids, desc="[Collect Product Images]"):
            assert isinstance(prod_id, int)
            imgs = self.read_prod_img(prod_id, save_only_first_image)
            if isinstance(imgs, list) and len(imgs) > 0:
                for idx, img in enumerate(imgs):
                    if pre_resize:
                        img = cv2.resize(img, dsize=(512, 512))
                    cv2.imwrite(
                        os.path.join(
                            self.prod_img_save_dir, f"{prod_id}_{idx:0>2d}.jpg"
                        ),
                        img,
                    )
            elif imgs is not None:
                if pre_resize:
                    imgs = cv2.resize(imgs, dsize=(512, 512))
                cv2.imwrite(
                    os.path.join(self.prod_img_save_dir, f"{prod_id}.jpg"), imgs
                )
            else:
                warnings.warn(f"Product is not found on the website: {prod_id}")

        interval = time() - tt
        print(
            f"Time: {interval:.2f} seconds\n", f"Saved in: {self.prod_img_save_dir}\n"
        )

    def compose_raw_data(self, return_data: bool = False):
        """수집한 데이터를 최종 검증하고 학습에 활용할 데이터셋을 구성. 기획전별 수집되지 않은 상품에 대한 파악 및 필터링 등을 포함"""
        print("[+] Compose Raw Train Data", end="\n ")
        tt = time()

        # load plan/prod data and get useful features with renaming to merge them distinctively
        plan_data = pd.read_csv(os.path.join(self.plan_save_dir, "plan_data_all.csv"))
        prod_data = pd.read_csv(os.path.join(self.prod_save_dir, "prod_data_all.csv"))

        plan_data = plan_data[self.plan_cols].rename(PLAN_RENAME_DICT, axis=1)
        prod_data = prod_data[self.prod_cols].rename(PROD_RENAME_DICT, axis=1)
        prod_data_dict = prod_data.set_index("prod_id").to_dict("index")

        unique_plan_ids = dict()  # {"고유ID": "기획전ID"}
        unique_prod_ids = dict()  # {"고유ID": "상품ID"}
        plan_avail_prods_num = dict()  # {'기획전ID': 수집 가능한 상품 수}
        plan_total_prods_num = dict()  # {'기획전ID': 기획전에 등록된 상품 수}

        for plan_id in tqdm(
            plan_data["plan_id"].tolist(), desc="[Verify Data Availability]"
        ):
            avail_prod_ids = set()
            total_prod_ids = set()
            sections = self.read_plan_section_info(plan_id, self.plan_section_meta_dir)

            for sec in sections:
                prod_ids = sec["productIds"]
                total_prod_ids.update(prod_ids)

                for prod_id in prod_ids:
                    # pass product that does not have review/zzim/sale count, prod_opendate
                    if prod_id not in prod_data_dict:
                        continue

                    # pass product that does not have image
                    elif not os.path.isfile(
                        os.path.join(self.prod_img_save_dir, f"{prod_id}.jpg")
                    ):
                        continue

                    elif (
                        prod_data_dict[prod_id]["prod_review_cnt"] is np.nan
                        or prod_data_dict[prod_id]["prod_zzim_cnt"] is np.nan
                        or prod_data_dict[prod_id]["prod_sale_cnt"] is np.nan
                        or prod_data_dict[prod_id]["prod_opendate"] is np.nan
                    ):
                        continue

                    elif os.path.isfile(
                        os.path.join(self.prod_img_meta_dir, f"{prod_id}.pkl")
                    ):
                        unique_id = f"{prod_id:0>10d}-{plan_id:0>6d}"
                        unique_plan_ids[unique_id] = plan_id
                        unique_prod_ids[unique_id] = prod_id
                        avail_prod_ids.add(prod_id)

            plan_avail_prods_num[plan_id] = len(avail_prod_ids)
            plan_total_prods_num[plan_id] = len(total_prod_ids)

        unique_plan_ids = pd.Series(unique_plan_ids).reset_index()
        unique_plan_ids.columns = ["id", "plan_id"]
        unique_prod_ids = pd.Series(unique_prod_ids).reset_index()
        unique_prod_ids.columns = ["id", "prod_id"]

        plan_avail_prods_num = pd.Series(plan_avail_prods_num).reset_index()
        plan_avail_prods_num.columns = ["plan_id", "plan_avail_prod_num"]
        plan_total_prods_num = pd.Series(plan_total_prods_num).reset_index()
        plan_total_prods_num.columns = ["plan_id", "plan_total_prod_num"]
        plan_prods_num = plan_avail_prods_num.merge(
            plan_total_prods_num, how="left", on="plan_id"
        )

        # 다음 조건을 만족하는 기획전만 수집
        # (1) 접근 가능한 상품 수가 50개 이상
        # 원래 전시 상품 수 대비 접근 가능한 상품 수 비율이 50% 이상
        plan_prods_num["plan_capacity"] = (
            plan_prods_num["plan_avail_prod_num"]
            / plan_prods_num["plan_total_prod_num"]
        )
        useful_plan_ids = plan_prods_num[
            (plan_prods_num["plan_avail_prod_num"] >= 50)
            & (plan_prods_num["plan_capacity"] >= 0.5)
        ]["plan_id"].tolist()

        # 학습에 활용할 만한 기획전만으로 raw 데이터 구성
        raw = unique_prod_ids.merge(unique_plan_ids, how="left", on="id")
        raw = raw.merge(plan_prods_num, how="left", on="plan_id")
        raw = raw[raw["plan_id"].isin(useful_plan_ids)]

        # merge plan/prod info
        raw = raw.merge(plan_data, how="left", on="plan_id")
        raw = raw.merge(prod_data, how="left", on="prod_id")

        print(
            "[+] Refined Data Description\n",
            f"# Plans Before Refinement: {len(unique_plan_ids):,d}\n",
            f"# Plans After Refinement: {len(useful_plan_ids):,d}\n",
            f"Prod Count for Each Category:\n",
        )

        # save
        raw.to_csv(os.path.join(self.parent_dir, "raw.csv"), index=False)

        interval = time() - tt
        print(f"Time: {interval:.2f} seconds\n", f"Saved in: {self.parent_dir}\n")

        if return_data:
            return raw

    def read_prod_img(self, prod_id: int, first: bool = True):
        meta_path = os.path.join(self.prod_img_meta_dir, f"{prod_id}.pkl")
        if not os.path.isfile(meta_path):
            return None

        images = load_pickle(meta_path)

        if first:  # attId = 1
            img = self.read_img(images[1]["imageUrl"])
            return img

        else:
            imgs = [
                self.read_img(url)
                for url in [images[i]["imageUrl"] for i in range(1, len(images) + 1)]
            ]
            return imgs

    def make_dirs(self, save_dir, start_date, end_date):
        self.parent_dir = os.path.join(save_dir, f"{start_date}_{end_date}")
        self.plan_save_dir = os.path.join(self.parent_dir, "plan_data")
        self.prod_save_dir = os.path.join(self.parent_dir, "prod_data")
        self.plan_section_meta_dir = os.path.join(
            self.plan_save_dir, "plan_section_meta"
        )
        self.prod_img_meta_dir = os.path.join(self.prod_save_dir, "prod_img_meta")
        self.prod_ocr_meta_dir = os.path.join(self.prod_save_dir, "prod_ocr_meta")
        self.prod_img_save_dir = os.path.join(self.prod_save_dir, "images")

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(self.parent_dir, exist_ok=True)
        os.makedirs(self.plan_save_dir, exist_ok=True)
        os.makedirs(self.prod_save_dir, exist_ok=True)
        os.makedirs(self.plan_section_meta_dir, exist_ok=True)
        os.makedirs(self.prod_img_meta_dir, exist_ok=True)
        os.makedirs(self.prod_ocr_meta_dir, exist_ok=True)
        os.makedirs(self.prod_img_save_dir, exist_ok=True)

    @staticmethod
    def digits2datetime(x):
        dt = datetime.fromtimestamp(int(x) / 1000)
        return dt.strftime("%Y-%m-%d")

    @staticmethod
    def read_plan_section_info(plan_id: int, plan_section_meta_dir: str):
        fpath = os.path.join(plan_section_meta_dir, f"{plan_id}.pkl")
        if not os.path.isfile(fpath):
            return None
        sections = load_pickle(fpath)
        return sections

    @staticmethod
    def refine_prods_meta(prods_table: pd.DataFrame):
        # remove products that removed or stopped sales
        prods_table = prods_table[prods_table["deleted"] == False]

        # make zzim, sale, discounted price unique
        max_zzims_sales = prods_table.groupby("prod_id", as_index=False)[
            ["zzimCount", "saleCount"]
        ].max()
        min_discounts_opens = prods_table.groupby("prod_id", as_index=False)[
            ["discountedSalePrice", "openDate"]
        ].min()
        prods_table = prods_table.drop(["zzimCount", "saleCount"], axis=1).merge(
            max_zzims_sales, how="left", on="prod_id"
        )
        prods_table = prods_table.drop(
            ["discountedSalePrice", "openDate"], axis=1
        ).merge(min_discounts_opens, how="left", on="prod_id")

        # remove overlapped
        prods_table = (
            prods_table.groupby("prod_id", as_index=False)
            .first()
            .reset_index(drop=True)
        )
        prods_table["prod_id"] = prods_table["prod_id"].astype(int)
        return prods_table

    @staticmethod
    def read_img(img_url):
        img_arr = np.array(bytearray(requests.get(img_url).content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return img

    @staticmethod
    def read_plans() -> Dict[int, Dict[str, int]]:
        """현재까지의 모든 기획전 정보를 수집"""
        date = datetime.today().strftime("%Y-%m-%d")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json; charset=utf-8",
        }
        data = {
            "namespace": "pct",
            "path": f"/user/shopping-vertical/export/plan/vertical-plan-{date}",
        }
        url = "http://c3s-hive-delegator.flova-pipeline.svc.ar2.io.navercorp.com:10080/api/hdfs/download"

        res = requests.post(url, headers=headers, data=json.dumps(data))
        res_content = res.content
        res_decode = res_content.decode("utf-8")
        res_list = res_decode.split("\n")  # 기획전 단위로 split됨
        res_json = [json.loads(json.loads(elt)) for elt in res_list[:-1]]

        plans = {}
        for elt in res_json:
            plans[elt["id"]] = elt
        return plans

    @staticmethod
    def get_year_month_day(date: str = None):  # like '2021-09-03'
        if date is None:
            today = datetime.today()
            year, month, day = today.year, today.month, today.day
        else:
            year, month, day = date.split("-")
            year = int(year)
            month = int(month)
            day = int(day)
        return year, month, day

    @staticmethod
    def extract_date(x):
        year, month, day = x["date"].values()
        date = f"{year}-{month:0>2d}-{day:0>2d}"
        return date


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", default="2021-08-05")
    parser.add_argument("--end_date", default="2021-08-06")
    parser.add_argument("--save_dir", default="../data/plan_data/")
    args = parser.parse_args()

    collector = PlanDataCollector()
    collector.collect(args.start_date, args.end_date, args.save_dir)
