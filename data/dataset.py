import warnings
from abc import *
from pathlib import Path
from typing import List, Union, Dict, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing import TextPreprocessor, remap_exhibit_keys, remap_prod_keys

ExhibitionId = int
ProductId = int

Topic = Dict[str, torch.Tensor]
RawTopic = Dict[str, str]
Image = torch.Tensor
RawImage = np.ndarray

DEFAULT_EXHIBIT_ATTRS = [
    'exhibit_name',
    'exhibit_startdate',
    'exhibit_cat1',
    'exhibit_cat2',
    'exhibit_kwds',
]
DEFAULT_PROD_ATTRS = [
    "prod_name",
    "prod_text",
    "prod_opendate",
    "prod_cat1",
    "prod_cat2",
    "prod_cat3",
    "prod_cat4",
    "prod_page_title",
]

SAMPLING_METHOD = [
    'weighted',
    'random',
    'sequential'
]


class ExhibitionDataset(Dataset, metaclass=ABCMeta):
    """네이버 쇼핑 기획전 전용 데이터셋

    <기획전 메타 데이터 구성 (column 구성)>
    I. 기획전 attribute (exhibit_attrs)
    - exhibit_id (int): 기획전 ID
    - exhibit_name (str): 기획전 제목
    - exhibit_startdate (str): 기획전 게시 날짜
    - exhibit_cat1 (str): 기획전 카테고리 (depth=1)
    - exhibit_cat2 (str): 기획전 카테고리 (depth=2)
    - exhibit_kwds (str): 기획전 키워드

    II. 상품 attribute (prod_attrs)
    - prod_id: (int): 상품 ID
    - prod_text (str): 상품 상세
    - prod_opendate (str): 상품 게시 날짜
    - prod_cat1 (str): 상품 카테고리 (depth=1)
    - prod_cat2 (str): 상품 카테고리 (depth=2)
    - prod_cat3 (str): 상품 카테고리 (depth=3)
    - prod_cat4 (str): 상품 카테고리 (depth=4)
    - prod_page_title (str): 상품 상세 페이지 제목

    III. Target: 랭킹 기준으로 삼을 점수
    - type 1. CTR: 상품별 클릭률
    - type 2. prod_review_cnt: 상품별 리뷰 수
    """

    def __init__(
            self,
            meta: pd.DataFrame,
            labeling_criterion: str,
            img_dir: str,
            img_transforms: A.Compose,
            txt_preprocessor: TextPreprocessor,
            exhibit_attrs: List[str] = None,
            prod_attrs: List[str] = None,
            sampling_method: str = 'weighted',  # 'weighted', 'random', 'sequential'
    ):
        """

        Args:
            meta (pd.DataFrame): 메타 데이터
            target (str): compatibility target
            img_dir (str): directory of product images
            exhibit_attrs (List[str], optional): exhibit attributes used to train. Defaults to None.
            prod_attrs (List[str], optional): product attributes used to train. Defaults to None.
                                              NOTE. it can be used for text augmentation
            sampling_method (str, optional): [description]. Defaults to 'weighted'.
        """
        if labeling_criterion not in meta.columns:
            raise ValueError(f"'{labeling_criterion}' column is not in meta")
        if sampling_method not in SAMPLING_METHOD:
            raise ValueError(f'sampling_method ({sampling_method}) should be one of {SAMPLING_METHOD}')

        if exhibit_attrs is None:
            exhibit_attrs = DEFAULT_EXHIBIT_ATTRS
        if prod_attrs is None:
            prod_attrs = DEFAULT_PROD_ATTRS

        if not set(exhibit_attrs).issubset(meta.columns.tolist()):
            raise ValueError('There exists a column in exhibit_attrs which is not in meta data.')
        if not set(prod_attrs).issubset(meta.columns.tolist()):
            raise ValueError('There exists a column in prod_attrs which is not in meta data.')

        necessary_columns = ['id', 'prod_id', 'exhibit_id', labeling_criterion] + exhibit_attrs + prod_attrs
        self.meta = meta[sorted(set(necessary_columns))]
        self.meta_by_exhibit_id: Dict[ExhibitionId, pd.DataFrame] = dict(tuple(meta.groupby('exhibit_id')))
        self.exhibit_ids = self.meta['exhibit_id'].unique().tolist()

        self.img_transforms = img_transforms
        self.txt_preprocessor = txt_preprocessor
        self.exhibit_attrs = exhibit_attrs
        self.prod_attrs = prod_attrs
        self.img_dir: Path = Path(img_dir)
        self.labeling_criterion = labeling_criterion
        self.sampling_method = sampling_method

        self.verify_meta_data()

    @abstractmethod
    def __getitem__(self, idx: int):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    def load_prod_img(self, prod_id: int) -> Tuple[RawImage, Path]:
        img_path: Path = self.img_dir / f'{prod_id}.jpg'
        img: RawImage = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        return img, img_path

    def verify_meta_data(self) -> bool:
        cannot_found_samples = self.meta[
            self.meta['prod_id'].apply(lambda x: not (self.img_dir / f'{x}.jpg').is_file())
        ]
        if len(cannot_found_samples) > 0:
            num_rows_origin = self.__len__()
            row_indices: List[int] = cannot_found_samples.index.tolist()
            self.meta = self.meta.drop(row_indices, axis=0).reset_index(drop=True)
            num_rows_dropped = self.__len__()

            warnings.warn(
                'There exist samples whose images are not found. Dropped them to train properly. '
                f'Data Size: {num_rows_origin:,d} -> {num_rows_dropped:,d}'
            )
            return False
        return True

    def sample_prods(self, exhibit: pd.DataFrame, n: int = 1) -> Union[pd.Series, pd.DataFrame]:
        """한 기획전으로부터 n개 샘플을 추출

        Args:
            exhibit (pd.DataFrame): 한 기획전에 대한 데이터프레임
            n (int, optional): [description]. Defaults to 1.

        Returns:
            Union[pd.Series, pd.DataFrame]: [description]
        """
        if n < 0:
            raise ValueError(f'n({n}) should be natural number.')

        if n == -1:
            prods: pd.DataFrame = exhibit.sort_values(by=self.labeling_criterion, ascending=False, ignore_index=True)
        elif n == 1:
            prods: pd.Series = self._sample_prod(exhibit)
        else:
            prods: pd.DataFrame = self._sample_prods(exhibit, n)
        return prods

    def _sample_prod(self, exhibit) -> pd.Series:
        if self.sampling_method == 'random':
            prod: pd.Series = exhibit.sample(1).squeeze()
        elif self.sampling_method == 'weighted':
            # +eps(1e-8): to operate weighted sampling properly
            exhibit['weight'] = (
                    exhibit[self.labeling_criterion].values + np.random.uniform(size=len(exhibit)) * 1e-8
            )
            # since result was not good if n=1, so sample positive with n=k iterations
            # NOTE. it is not that time consuming(+-1ns)
            prod: pd.DataFrame = exhibit.sample(n=7, weights='weight')
            prod = prod[prod['weight'] == prod['weight'].max()].drop('weight', axis=1)
            prod: pd.Series = prod.sample(1).squeeze()
        elif self.sampling_method == 'sequential':
            prod: pd.DataFrame = exhibit[
                exhibit[self.labeling_criterion] == exhibit[self.labeling_criterion].max()
                ]
            prod: pd.Series = prod.sample(1).squeeze()
        else:
            raise NotImplementedError
        return prod

    def _sample_prods(self, exhibit: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(exhibit) == n:
            prods: pd.DataFrame = exhibit.sort_values(
                by=self.labeling_criterion, ascending=False, ignore_index=True
            )
        elif self.sampling_method == 'random':
            prods: pd.DataFrame = exhibit.sample(n=n, replace=False)
            if self.labeling_criterion in prods.columns:
                prods: pd.DataFrame = prods.sort_values(by=self.labeling_criterion, ascending=False, ignore_index=True)
            else:
                prods: pd.DataFrame = prods.reset_index(drop=True)
        elif self.sampling_method == 'weighted':
            exhibit['weight'] = (
                    exhibit[self.labeling_criterion].values + np.random.uniform(size=len(exhibit)) * 1e-8
            )
            pos_prod: pd.DataFrame = exhibit.sample(n=5, weights='weight')
            pos_prod: pd.DataFrame = pos_prod[pos_prod['weight'] == pos_prod['weight'].max()].sample(1)

            exhibit_except_pos: pd.DataFrame = exhibit.drop(pos_prod.index.item(), axis=0)
            neg_prods: pd.DataFrame = exhibit_except_pos.sample(
                n=n - 1,
                weights=exhibit_except_pos['weight'].max() - exhibit_except_pos['weight'],
            )
            prods: pd.DataFrame = pd.concat([pos_prod, neg_prods], axis=0)
            prods: pd.DataFrame = (prods.sort_values(by='weight', ascending=False, ignore_index=True)
                                   .drop('weight', axis=1))
        elif self.sampling_method == 'sequential':
            exhibit: pd.DataFrame = exhibit.sort_values(
                by=self.labeling_criterion, ascending=False, ignore_index=True
            )
            prods: pd.DataFrame = exhibit.head(n)
        else:
            raise NotImplementedError
        return prods


class TopicMatchingDataset(ExhibitionDataset):
    def __init__(
            self,
            meta: pd.DataFrame,
            labeling_criterion: str,
            img_dir: str,
            img_transforms: A.Compose,
            txt_preprocessor: TextPreprocessor,
            exhibit_attrs: List[str] = None,
            prod_attrs: List[str] = None,
            sampling_method: str = 'random',
            txt_aug_prob: float = 0.3,  # 텍스트를 상품 자체 텍스트 정보로 교체할 확률
    ):
        super(TopicMatchingDataset, self).__init__(
            meta=meta,
            labeling_criterion=labeling_criterion,
            img_dir=img_dir,
            img_transforms=img_transforms,
            txt_preprocessor=txt_preprocessor,
            exhibit_attrs=exhibit_attrs,
            prod_attrs=prod_attrs,
            sampling_method=sampling_method,
        )
        self.p = txt_aug_prob

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, exhibit_id: int) -> Tuple[Topic, Image]:
        if exhibit_id not in self.exhibit_ids:
            raise ValueError(f'exhibit_id ({exhibit_id}) is not in meta data')
        # sample product with topic info from a group
        group_sample: pd.Series = self.sample_prods(self.meta_by_exhibit_id[exhibit_id])

        # Topics
        #   - Basically, we use text of exhibiton topic.
        #   - But if we apply text augmentation, product's text description will be additionally used.
        if (self.prod_attrs is not None) and (np.random.binomial(1, p=self.p)):
            txt_raw: RawTopic = group_sample[self.prod_attrs].to_dict()
            topics: Topic = self.txt_preprocessor(**remap_prod_keys(txt_raw))
        else:
            txt_raw: RawTopic = group_sample[self.exhibit_attrs].to_dict()
            topics: Topic = self.txt_preprocessor(**remap_exhibit_keys(txt_raw))

        # Images
        images, _ = self.load_prod_img(group_sample['prod_id'])
        images: Image = self.img_transforms(image=images)['image']
        return topics, images


class ImageRankingDataset(ExhibitionDataset):
    def __init__(
            self,
            meta: pd.DataFrame,
            labeling_criterion: str,
            img_dir: str,
            img_transforms: A.Compose,
            txt_preprocessor: TextPreprocessor,
            exhibit_attrs: List[str],
            sampling_method: str = 'weighted',
            group_sampling_size: int = 30,  # K
    ):
        if group_sampling_size <= 1:
            raise ValueError(f'group_sampling_size ({group_sampling_size}) should be greater than 1.')

        super(ImageRankingDataset, self).__init__(
            meta=meta,
            labeling_criterion=labeling_criterion,
            img_dir=img_dir,
            img_transforms=img_transforms,
            txt_preprocessor=txt_preprocessor,
            exhibit_attrs=exhibit_attrs,
            prod_attrs=None,
            sampling_method=sampling_method,
        )
        self.K = group_sampling_size

    def __len__(self):
        return len(self.exhibit_ids)

    def __getitem__(self, exhibit_id: int) -> Tuple[Topic, Image]:
        if exhibit_id not in self.exhibit_ids:
            raise ValueError(f'exhibit_id ({exhibit_id}) is not in meta data')
        group: pd.DataFrame = self.meta_by_exhibit_id[exhibit_id].reset_index(drop=True)

        # Topic
        txt_raw: RawTopic = group.head(1)[self.exhibit_attrs].squeeze().to_dict()
        topic: Topic = self.txt_preprocessor(**remap_exhibit_keys(txt_raw))

        # Images
        prods: pd.DataFrame = self.sample_prods(group, n=self.K)
        images: List[Image] = []
        for _, prod in prods.iterrows():
            image, _ = self.load_prod_img(prod['prod_id'])
            images.append(self.img_transforms(image=image)['image'])
        images: Image = torch.stack(images)
        return topic, images


class TopicMatchingEvalDataset(ExhibitionDataset):
    def __init__(
            self,
            meta: pd.DataFrame,
            labeling_criterion: str,
            img_dir: str,
            img_transforms: A.Compose,
            txt_preprocessor: TextPreprocessor,
            exhibit_attrs: List[str] = None,
            prod_attrs: List[str] = None,
            sampling_method: str = 'random',
    ):
        super(TopicMatchingEvalDataset, self).__init__(
            meta=meta,
            labeling_criterion=labeling_criterion,
            img_dir=img_dir,
            img_transforms=img_transforms,
            txt_preprocessor=txt_preprocessor,
            exhibit_attrs=exhibit_attrs,
            prod_attrs=prod_attrs,
            sampling_method=sampling_method,
        )

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, exhibit_id: int) -> Tuple[Topic, Image, Dict[str, Union[int, float, RawTopic]]]:
        if exhibit_id not in self.exhibit_ids:
            raise ValueError(f'exhibit_id ({exhibit_id}) is not in meta data')
        desc = dict(
            id=0,
            exhibit_id=0,
            exhibit_attrs={a: None for a in self.exhibit_attrs},
            prod_id=0,
            prod_criterion_value=0.,
            prod_image_path='',
        )  # descriptions

        group_sample: pd.Series = self.sample_prods(self.meta_by_exhibit_id[exhibit_id])

        txt_raw: RawTopic = group_sample[self.exhibit_attrs].to_dict()
        topic: Topic = self.txt_preprocessor(**remap_exhibit_keys(txt_raw))

        image, image_path = self.load_prod_img(group_sample['prod_id'])
        image: Image = self.img_transforms(image=image)['image']

        desc['id']: int = group_sample['id']
        desc['exhibit_id']: ExhibitionId = exhibit_id
        desc['exhibit_attrs']: RawTopic = txt_raw
        desc['prod_id']: ProductId = group_sample['prod_id']
        desc["prod_criterion_value"]: float = group_sample[self.labeling_criterion]
        desc["prod_image_path"] = str(image_path)

        return topic, image, desc


class ImageRankingEvalDataset(ExhibitionDataset):
    def __init__(
            self,
            meta: pd.DataFrame,
            labeling_criterion: str,
            img_dir: str,
            img_transforms: A.Compose,
            txt_preprocessor: TextPreprocessor,
            exhibit_attrs: List[str],
            sampling_method: str = 'weighted',
            group_sampling_size: int = 30,
    ):
        super(ImageRankingEvalDataset, self).__init__(
            meta=meta,
            labeling_criterion=labeling_criterion,
            img_dir=img_dir,
            img_transforms=img_transforms,
            txt_preprocessor=txt_preprocessor,
            exhibit_attrs=exhibit_attrs,
            prod_attrs=None,
            sampling_method=sampling_method,
        )
        self.K = group_sampling_size

    def __len__(self) -> int:
        return len(self.exhibit_ids)

    def __getitem__(self, exhibit_id: int) -> Tuple[Topic, Image, Dict[str, Union[int, float, RawTopic]]]:
        if exhibit_id not in self.exhibit_ids:
            raise ValueError(f'exhibit_id ({exhibit_id}) is not in meta data')
        desc = dict(
            id=[],
            exhibit_id=0,
            exhibit_attrs={a: None for a in self.exhibit_attrs},
            prod_id=[],
            prod_criterion_value=[],
            prod_image_path=[],
        )
        desc['exhibit_id']: ExhibitionId = exhibit_id
        group: pd.DataFrame = self.meta_by_exhibit_id[exhibit_id].reset_index(drop=True)

        txt_raw: RawTopic = group.head(1)[self.exhibit_attrs].squeeze().to_dict()
        topic: Topic = self.txt_preprocessor(**remap_exhibit_keys(txt_raw))
        desc['exhibit_attrs']: RawTopic = txt_raw

        prods: pd.DataFrame = self.sample_prods(group, n=self.K)
        images: List[Image] = []
        for _, row in prods.iterrows():
            prod_id: ProductId = row['prod_id']
            image, image_path = self.load_prod_img(prod_id)
            images.append(self.img_transforms(image=image)['image'])
            desc['id'].append(row['id'])
            desc['prod_id'].append(prod_id)
            desc["prod_criterion_value"].append(row[self.labeling_criterion])
            desc["prod_image_path"].append(image_path)
        images: Image = torch.stack(images)
        return topic, images, desc
