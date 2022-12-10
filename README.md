# CLIK: Contrastive Learning for Topic-Dependent Image Ranking
* Paper Link: [Jihyeong Ko, Contrastive Learning for Topic-Dependent Image Ranking](https://drive.google.com/file/d/1p0GVZzbrjA6_pqpmQYrrS6R_l_WBicim/view)
* Published(expected): [The 4th Workshop on RecSys 2022 in Fashion & Retail, *fashionXrecsys*](https://recsys.acm.org/recsys22/fashionxrecsys/)

## Abstract
We propose a novel method **CLIK (Contrastive Learning for topic-dependent Image ranKing)** that selects the best from multiple images considering a topic. It can consider two factors for the selection: (1) *how attractive each image is to users* and (2) *how well each image fits the given product concept (i.e. topic)*. To understand both factors simultaneously, CLIK performs two novel training tasks. At first, in ***Topic Matching***, it learns the semantic relationship between various images and topics based on contrastive learning. Secondly, in ***Image Ranking***, it ranks given images considering a given topic leveraging knowledge learned from ***Topic Matching*** using contrastive loss. Both training tasks are done simultaneously by integrated modules with shared weights. CLIK showed significant offline evaluation results and had more positive feedback from users in online A/B testing compared to existing methods.

## Method
![](https://github.com/iloveslowfood/CLIK/blob/master/etc/CLIK02.png?raw=true)

![](https://github.com/iloveslowfood/CLIK/blob/master/etc/CLIK01.png?raw=true)

## Experiment
![](https://github.com/iloveslowfood/CLIK/blob/master/etc/CLIK03.png?raw=true)

![](https://github.com/iloveslowfood/CLIK/blob/master/etc/CLIK04.png?raw=true)

![](https://github.com/iloveslowfood/CLIK/blob/master/etc/CLIK05.png?raw=true)

## Data Collection
* 모델 학습을 위한 데이터는 [네이버쇼핑 기획전 서비스](https://shopping.naver.com/plan2/p/index.naver)로부터 수집됩니다.
* 구체적인 수집 방법 및 절차는 [[Guide] Data Collection.ipynb](https://github.com/iloveslowfood/CLIK/blob/main/etc/%5BGuide%5D%20Data%20Collection.ipynb)에서 확인하실 수 있습니다.


---
## Usage

### Dependencies
* Python >= 3.6
* CUDA == 11.1
* torch == 1.9.1

### Train

#### Single-gpu Training
* 싱글 GPU
```shell
$ nsml run -m '[SESS_DESC]' -e main.py -a '--config_path ./configs/CLIK.yaml' -d [DATA] --memory '[MEM]' \
  -g 1 -c [NUM_CPUs] --gpu-driver-version 455.32 --gpu-model [GPU_TYPE]
```
* 실제 구동 예시
```shell
$ nsml run -m 'CLIK(128/10/4)' -e main.py -a '--config_path ./configs/CLIK.yaml' -d vplan_ver_2-2 \
  --memory '15G' -g 1 -c 2 --gpu-driver-version 455.32 --gpu-model P40
```

#### Distributed Training
* 분산학습
```shell
$ nsml run -m '[SESS_DESC]' -e main.py -a '--config_path ./configs/CLIK.yaml' -d [DATA] --memory '[MEM]' \
  -g [NUM_GPUs] -c [NUM_CPUs]--gpu-driver-version 455.32 --gpu-model [GPU_TYPE] --shm-size '[SHARED_MEM]'
```
* 실제 구동 예시
```
$ nsml run -m 'CLIK(512/20/12)' -e main.py -a '--config_path ./configs/CLIK.yaml' -d vplan_ver_2-2 \
  --memory '24G' -g 4 -c 4 --gpu-driver-version 455.32 --gpu-model P40 --shm-size '4G'
```

### Inference
`TaskOperator` 를 통해 간편한 모델 추론이 가능합니다. 학습한 모델 및 텍스트/이미지 전처리 도구를 입력해 `TaskOperator` 를 초기화한 뒤, 적절한 기획전/상품 정보를 입력하여 대표 이미지를 선출합니다.
```python
import torch
from networks import CLIK
from preprocessing import get_eval_transforms, TextPreprocessor
from inference import TaskOperator

device = torch.device('cpu')
model = CLIK(feature_dim=128, queue_size=512)
img_transforms = get_eval_transforms(224, 224)
txt_preprocessor = TextPreprocessor()

# 초기화
operator = TaskOperator(
    model,                      # 학습한 모델
    img_transforms,             # 이미지 전처리기
    txt_preprocessor            # 텍스트 전처리기
    device=device
)

# 추론 형식
best_product_urls = operator(
    plan_attrs=[기획전정보],      # 기획전 텍스트 정보 (Dict[str, str])
    prod_urls=[상품이미지URLs],   # 상품 URL (또는 이미지 어레이) (Union[List[str], List[np.array]])
    topk=1                      # 상위 K개 추천 결과를 리턴 (int)
)
```

추론 예시는 다음과 같습니다.
```python
plan_attrs = {
    'name': '주문폭주 가을코디 매일매일 업뎃',     # 기획전 제목
    'date': '2021-08-30',                    # 기획전 게시 시기(입력하지 않을 경우 오늘 날짜로 대체됩니다.
    'cat1': '패션',                           # 기획전 카테고리(depth=1)
    'cat2': '여성의류',                        # 기획전 카테고리(depth=2)
    'kwds': '가을신상,데일리룩,데이트룩,캐주얼룩'   # 기획전키워드(공백 없이 입력)
}

# 'http~' 형식의 이미지 url을 입력하셔도 됩니다
img_urls = [ 
    '../data/plan_data/2021-08-05_2021-10-05/prod_data/images/5812499204.jpg', 
    '../data/plan_data/2021-08-05_2021-10-05/prod_data/images/5817476543.jpg',
    ...
]
best_urls = operator(plan_attrs, img_urls, topk=3) # 모델에 입력하여 베스트 상품 이미지 url을 추출
print(best_urls)
------------------------------------------------------------------------------------------
['../data/plan_data/2021-08-05_2021-10-05/prod_data/images/5812499204.jpg', 
 '../data/plan_data/2021-08-05_2021-10-05/prod_data/images/5817476543.jpg',
 '../data/plan_data/2021-08-05_2021-10-05/prod_data/images/5814574744.jpg']
```
기획전의 attributes는 모델에 다음과 같은 형태로 입력합니다. Key에 적절한 텍스트가 입력되지 않을 경우 공백 처리가 되어 모델에 입력되며, `"date"`의 경우 미입력시 오늘 날짜가 입력됩니다.
```python
plan_attrs = {
    "name": [기획전명],
    "desc": [기획전 상세],
    "date": [기획전 게시날짜],
    "cat1": [기획전 카테고리 (depth=1)],
    "cat2": [기획전 카테고리 (depth=2)],
    "cat3": [기획전 카테고리 (depth=3)], # 해당 key는 text augmentation에 활용되며, 추론 단계에서는 사용되지 않습니다.
    "cat4": [기획전 카테고리 (depth=4)], # 해당 key는 text augmentation에 활용되며, 추론 단계에서는 사용되지 않습니다.
    "kwds": [기획전 키워드],
}
```

---
## Evaluation
* [네이버쇼핑 기획전 서비스](https://shopping.naver.com/plan2/p/index.naver) 데이터 기반 모델 평가 결과입니다.
* `Data Type I` - 학습 기획전 수: 3.7K, 학습 상품 수: 2.6M
  * 타깃: 리뷰수
  * 수집 기간: 2018. 01. 01 \~ 2021. 12. 31
* `Data Type II` - 학습 기획전 수: 1.2K, 학습 상품 수: 83K
  * 타깃: CTR
  * 수집 기간: 2021. 08. 05 \~ 2021. 11. 05
  
![](https://github.com/iloveslowfood/CLIK/blob/main/etc/evaluation_results.jpg?raw=true)

---
## Code Format
black == 21.9b0
