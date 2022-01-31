import re
from typing import List
from datetime import datetime
import math
import numpy as np
import torch
from transformers import BertTokenizer

# season intervals
SPRING = ("01-01", "03-31")
SUMMER = ("04-01", "06-30")
AUTUMN = ("07-01", "09-30")
WINTER = ("10-01", "12-31")

PLAN_ATTR_MAPPER = {
    "plan_name": "name",
    "plan_page_title": "desc",
    "plan_startdate": "date",
    "plan_cat1": "cat1",
    "plan_cat2": "cat2",
    "plan_kwds": "kwds",
}

PROD_ATTR_MAPPER = {
    "prod_name": "name",
    "prod_text": "desc",
    "prod_opendate": "date",
    "prod_cat1": "cat1",
    "prod_cat2": "cat2",
    "prod_cat3": "cat3",
    "prod_cat4": "cat4",
    "prod_page_title": "kwds",
}


def make_kwd_tidy(x: str) -> str:
    return " ".join(list(map(remove_special_chars, x.split(","))))


def remove_special_chars(x: str) -> str:
    p = re.compile(r"[a-zA-Z0-9가-힇ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥]+")
    return " ".join(p.findall(x)).strip()


def remap_plan_keys(d):
    remapped = dict()
    for k, v in d.items():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        remapped[PLAN_ATTR_MAPPER[k]] = v
    return remapped


def remap_prod_keys(d):
    remapped = dict()
    for k, v in d.items():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        remapped[PROD_ATTR_MAPPER[k]] = v
    return remapped


class TextPreprocessor:
    """기획전/상품 텍스트 데이터 전처리 클래스"""

    def __init__(
        self,
        pretrained_tokenizer: str = "dsksd/bert-ko-small-minimal",
        max_length: int = 128,
        dropout: float = 0.1,
    ):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer)
        self.max_length = max_length
        self.dropout = dropout
        assert self.tokenizer.unk_token_id is not None

        # dict for mapping tokenization function
        self.__mapper = {
            "name": self.tokenize_sentence,
            "desc": self.tokenize_sentence,
            "date": self.tokenize_date,
            "cat1": self.tokenize_cat,
            "cat2": self.tokenize_cat,
            "cat3": self.tokenize_cat,
            "cat4": self.tokenize_cat,
            "kwds": self.tokenize_kwds,
        }

        # tokenization order
        self.__order = [
            "name",
            "desc",
            "date",
            "cat1",
            "cat2",
            "cat3",
            "cat4",
            "kwds",
        ]

    def __call__(self, **kwargs):
        return self.preprocess(**kwargs)

    def preprocess(self, expand_dim: bool = False, **kwargs):
        if set(kwargs.keys()) - set(self.__order):
            raise ValueError(
                f"Unexpected text attributes were used: {tuple(set(kwargs.keys()) - set(self.__order))}. Following attributes are only available:\n"
                "'name'(title), 'desc'(description), 'date'(opendate or today)\n"
                "'cat1'(category depth=1), 'cat2'(category depth=2), 'cat3'(category depth=3), 'cat4'(category depth=4)\n"
                "'kwds'(keywords info)"
            )

        encoded = {attr: [] for attr in self.__order}
        num_pads, num_special_tokens = 0, 9
        current_length = 0 + num_special_tokens

        for attr in self.__order:
            content = kwargs.get(attr, None)
            if content is not None:
                tokenize_fn = self.__mapper[attr]
                tokenized = tokenize_fn(content)

                if len(tokenized) == 0:  # pass if tokenized output is empty
                    continue

                if self.dropout > 0.0:  # dropout
                    tokenized = self.apply_dropout(tokenized, self.dropout)
                encoded[attr].extend(tokenized)
                current_length += len(tokenized)

        # Drop 우선순위: desc -> kwds -> name
        if current_length > self.max_length:
            num_drops = current_length - self.max_length

            if num_drops <= len(encoded["desc"]):
                encoded["desc"] = encoded["desc"][: len(encoded["desc"]) - num_drops]

            else:
                if "kwds" in encoded.keys():
                    num_drops -= len(encoded["desc"])
                    encoded["desc"] = []  # 키워드 모두 제거
                    if num_drops <= len(encoded["kwds"]):
                        encoded["kwds"] = encoded["kwds"][
                            : len(encoded["kwds"]) - num_drops
                        ]
                    else:
                        num_drops -= len(encoded["kwds"])
                        encoded["kwds"] = []
                        encoded["name"] = encoded["name"][
                            : len(encoded["name"]) - num_drops
                        ]
                else:
                    num_drops -= len(encoded["desc"])
                    encoded["desc"] = []  # 키워드 모두 제거
                    encoded["name"] = encoded["name"][
                        : len(encoded["name"]) - num_drops
                    ]

        elif current_length < self.max_length:
            num_pads = self.max_length - current_length

        # [CLS] 기획전명 [SEP] 기획전페이지명 [SEP] 계절 [SEP] 카테고리1 [SEP] 카테고리2 [SEP] 기획전키워드 [SEP] + 남은 부분 [PAD]
        input_ids = [self.tokenizer.cls_token_id]
        for attr in self.__order:
            input_ids.extend(encoded[attr])
            input_ids.append(self.tokenizer.sep_token_id)
        input_ids.extend([self.tokenizer.pad_token_id] * num_pads)

        # 기획전명: 0, 그 외 정보: 1, 패딩: 0
        token_type_ids = []
        num_special_tokens_plan_name = 2  # [CLS] 기획전명 [SEP]
        num_special_tokens_others = 1
        for attr in self.__order:
            if attr == "name":
                token_type_ids.extend(
                    [0] * (len(encoded[attr]) + num_special_tokens_plan_name)
                )
            else:
                token_type_ids.extend(
                    [1] * (len(encoded[attr]) + num_special_tokens_others)
                )
        token_type_ids.extend([0] * num_pads)

        # 관측값: 1, 패딩: 0
        attention_mask = [1] * (self.max_length - num_pads) + [0] * num_pads

        # check each input has proper length
        assert len(input_ids) == self.max_length
        assert len(token_type_ids) == self.max_length
        assert len(attention_mask) == self.max_length

        preprocessed = dict(
            input_ids=torch.tensor(input_ids),
            token_type_ids=torch.tensor(token_type_ids),
            attention_mask=torch.tensor(attention_mask),
        )

        if expand_dim:
            for k in preprocessed.keys():
                preprocessed[k] = preprocessed[k].unsqueeze(0)

        return preprocessed

    def tokenize_sentence(self, sentence: str) -> List[int]:
        sentence = remove_special_chars(sentence)
        sentence = self.tokenizer.encode(sentence, add_special_tokens=False)
        return sentence

    def tokenize_date(self, date: str) -> List[int]:
        year = date.split("-")[0]

        if (
            datetime.strptime(f"{year}-{SPRING[0]}", "%Y-%m-%d")
            <= datetime.strptime(date, "%Y-%m-%d")
            <= datetime.strptime(f"{year}-{SPRING[1]}", "%Y-%m-%d")
        ):
            season = "봄"

        elif (
            datetime.strptime(f"{year}-{SUMMER[0]}", "%Y-%m-%d")
            <= datetime.strptime(date, "%Y-%m-%d")
            <= datetime.strptime(f"{year}-{SUMMER[1]}", "%Y-%m-%d")
        ):
            season = "여름"

        elif (
            datetime.strptime(f"{year}-{AUTUMN[0]}", "%Y-%m-%d")
            <= datetime.strptime(date, "%Y-%m-%d")
            <= datetime.strptime(f"{year}-{AUTUMN[1]}", "%Y-%m-%d")
        ):
            season = "가을"

        elif (
            datetime.strptime(f"{year}-{WINTER[0]}", "%Y-%m-%d")
            <= datetime.strptime(date, "%Y-%m-%d")
            <= datetime.strptime(f"{year}-{WINTER[1]}", "%Y-%m-%d")
        ):
            season = "겨울"

        season = self.tokenizer.encode(season, add_special_tokens=False)
        return season

    def tokenize_cat(self, cat: str) -> List[int]:
        cat = self.tokenizer.encode(cat, add_special_tokens=False)
        return cat

    def tokenize_kwds(self, kwds: str) -> List[int]:
        kwds = make_kwd_tidy(kwds)
        kwds = self.tokenizer.encode(kwds, add_special_tokens=False)
        return kwds

    def apply_dropout(self, tokenized: List[int], prob: float) -> List[int]:
        dropout_mask = np.random.binomial(np.ones_like(tokenized), prob)
        tokenized_do = [
            t if dropout_mask[i] == 0 else self.tokenizer.unk_token_id
            for i, t in enumerate(tokenized)
        ]
        assert len(tokenized) == len(tokenized_do)
        return tokenized_do
