import math
import re
from datetime import datetime
from typing import List, Any, Dict

import numpy as np
import torch
from transformers import BertTokenizer

SPRING = ('01-01', '03-31')
SUMMER = ('04-01', '06-30')
AUTUMN = ('07-01', '09-30')
WINTER = ('10-01', '12-31')

EXHIBIT_ATTR_MAPPER = {
    'exhibit_name': 'name',
    'exhibit_page_title': 'desc',
    'exhibit_startdate': 'date',
    'exhibit_cat1': 'cat1',
    'exhibit_cat2': 'cat2',
    'exhibit_kwds': 'kwds',
}

PROD_ATTR_MAPPER = {
    'prod_name': 'name',
    'prod_text': 'desc',
    'prod_opendate': 'date',
    'prod_cat1': 'cat1',
    'prod_cat2': 'cat2',
    'prod_cat3': 'cat3',
    'prod_cat4': 'cat4',
    'prod_page_title': 'kwds',
}

TOKENS = List[int]


def make_kwd_tidy(x: str) -> str:
    return ' '.join(list(map(remove_special_chars, x.split(','))))


def remove_special_chars(x: str) -> str:
    p = re.compile(r'[a-zA-Z0-9가-힇ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥]+')
    return ' '.join(p.findall(x)).strip()


def remap_exhibit_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    remapped: Dict[str, Any] = {}
    for k, v in d.items():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        remapped[EXHIBIT_ATTR_MAPPER[k]] = v
    return remapped


def remap_prod_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    remapped: Dict[str, Any] = {}
    for k, v in d.items():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        remapped[PROD_ATTR_MAPPER[k]] = v
    return remapped


class TextPreprocessor:
    """기획전/상품 텍스트 데이터 전처리 클래스"""

    def __init__(
            self,
            pretrained_tokenizer: str = 'dsksd/bert-ko-small-minimal',
            max_length: int = 128,
            dropout: float = 0.1,
    ):
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer)
        self.max_length = max_length
        self.dropout = dropout
        self.__mapper = {
            'name': self._tokenize_sentence,
            'desc': self._tokenize_sentence,
            'date': self._tokenize_date,
            'cat1': self._tokenize_cat,
            'cat2': self._tokenize_cat,
            'cat3': self._tokenize_cat,
            'cat4': self._tokenize_cat,
            'kwds': self._tokenize_kwds,
        }
        self.__order = [
            'name',
            'desc',
            'date',
            'cat1',
            'cat2',
            'cat3',
            'cat4',
            'kwds',
        ]

    def __call__(self, **kwargs) -> Dict[str, torch.Tensor]:
        return self.preprocess(**kwargs)

    def preprocess(self, expand_dim: bool = False, **kwargs) -> Dict[str, torch.Tensor]:
        if set(kwargs.keys()) - set(self.__order):
            raise ValueError(
                f'Unexpected text attributes were used: {tuple(set(kwargs.keys()) - set(self.__order))}. '
                f'Following attributes are only available:\n'
                "'name'(title), 'desc'(description), 'date'(opendate or today)\n"
                "'cat1'(category depth=1), 'cat2'(category depth=2), 'cat3'(category depth=3), "
                "'cat4'(category depth=4)\n'kwds'(keywords info)"
            )

        encoded: Dict[str, TOKENS] = {attr: [] for attr in self.__order}
        num_pads = 0
        num_special_tokens = 9
        current_length = num_special_tokens

        for attr in self.__order:
            content: str = kwargs.get(attr, None)
            if content is not None:
                tokenize_fn = self.__mapper[attr]
                tokenized: TOKENS = tokenize_fn(content)
                if len(tokenized) == 0:
                    continue
                if self.dropout > 0.0:
                    tokenized: TOKENS = self._apply_dropout(tokenized, self.dropout)
                encoded[attr].extend(tokenized)
                current_length += len(tokenized)

        # drop (priority: desc -> kwds -> name)
        if current_length > self.max_length:
            num_drops: int = current_length - self.max_length
            if num_drops <= len(encoded['desc']):
                encoded['desc']: TOKENS = encoded['desc'][: len(encoded['desc']) - num_drops]
            else:
                if 'kwds' in encoded.keys():
                    num_drops -= len(encoded['desc'])
                    encoded['desc']: TOKENS = []  # 키워드 모두 제거
                    if num_drops <= len(encoded['kwds']):
                        encoded['kwds']: TOKENS = encoded['kwds'][: len(encoded['kwds']) - num_drops]
                    else:
                        num_drops -= len(encoded['kwds'])
                        encoded['kwds']: TOKENS = []
                        encoded['name']: TOKENS = encoded['name'][: len(encoded['name']) - num_drops]
                else:
                    num_drops -= len(encoded['desc'])
                    encoded['desc']: TOKENS = []  # 키워드 모두 제거
                    encoded['name']: TOKENS = encoded['name'][: len(encoded['name']) - num_drops]
        elif current_length < self.max_length:
            num_pads: int = self.max_length - current_length

        # input tokens
        # [CLS] name [SEP] desc [SEP] date [SEP] cat1 [SEP] cat2 [SEP] kwds [SEP] [PAD] ...
        input_ids: TOKENS = [self.tokenizer.cls_token_id]
        for attr in self.__order:
            input_ids.extend(encoded[attr])
            input_ids.append(self.tokenizer.sep_token_id)
        input_ids.extend([self.tokenizer.pad_token_id] * num_pads)

        # token_type_ids
        # name: 0, others: 1, padding: 0
        token_type_ids: TOKENS = []
        num_special_tokens_exhibit_name = 2  # [CLS] 기획전명 [SEP]
        num_special_tokens_others = 1
        for attr in self.__order:
            if attr == 'name':
                token_type_ids.extend([0] * (len(encoded[attr]) + num_special_tokens_exhibit_name))
            else:
                token_type_ids.extend([1] * (len(encoded[attr]) + num_special_tokens_others))
        token_type_ids.extend([0] * num_pads)

        # attention_mask
        # observed: 1, padding: 0
        attention_mask: TOKENS = [1] * (self.max_length - num_pads) + [0] * num_pads

        # output
        preprocessed: Dict[str, torch.Tensor] = dict(
            input_ids=torch.tensor(input_ids),
            token_type_ids=torch.tensor(token_type_ids),
            attention_mask=torch.tensor(attention_mask),
        )
        if expand_dim:
            for k in preprocessed.keys():
                preprocessed[k]: torch.Tensor = preprocessed[k].unsqueeze(0)
        return preprocessed

    def _tokenize_sentence(self, sentence: str) -> TOKENS:
        sentence: str = remove_special_chars(sentence)
        sentence: TOKENS = self.tokenizer.encode(sentence, add_special_tokens=False)
        return sentence

    def _tokenize_date(self, date: str) -> TOKENS:
        year: str = date.split('-')[0]
        if (
                datetime.strptime(f'{year}-{SPRING[0]}', '%Y-%m-%d')
                <= datetime.strptime(date, '%Y-%m-%d')
                <= datetime.strptime(f'{year}-{SPRING[1]}', '%Y-%m-%d')
        ):
            season = '봄'

        elif (
                datetime.strptime(f'{year}-{SUMMER[0]}', '%Y-%m-%d')
                <= datetime.strptime(date, '%Y-%m-%d')
                <= datetime.strptime(f'{year}-{SUMMER[1]}', '%Y-%m-%d')
        ):
            season = '여름'

        elif (
                datetime.strptime(f'{year}-{AUTUMN[0]}', '%Y-%m-%d')
                <= datetime.strptime(date, '%Y-%m-%d')
                <= datetime.strptime(f'{year}-{AUTUMN[1]}', '%Y-%m-%d')
        ):
            season = '가을'
        else:
            season = '겨울'
        season: TOKENS = self.tokenizer.encode(season, add_special_tokens=False)
        return season

    def _tokenize_cat(self, cat: str) -> TOKENS:
        cat: TOKENS = self.tokenizer.encode(cat, add_special_tokens=False)
        return cat

    def _tokenize_kwds(self, kwds: str) -> TOKENS:
        kwds: str = make_kwd_tidy(kwds)
        kwds: TOKENS = self.tokenizer.encode(kwds, add_special_tokens=False)
        return kwds

    def _apply_dropout(self, tokenized: TOKENS, prob: float) -> TOKENS:
        dropout_mask: np.ndarray = np.random.binomial(np.ones_like(tokenized), prob)
        tokenized_dropout: TOKENS = [
            t if dropout_mask[i] == 0 else self.tokenizer.unk_token_id
            for i, t in enumerate(tokenized)
        ]
        return tokenized_dropout
