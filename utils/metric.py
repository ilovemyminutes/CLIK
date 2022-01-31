from typing import List, Union
import numpy as np
import torch


def accuracy(similarities: torch.Tensor, labels: torch.Tensor) -> float:
    predicted = similarities.argmax(dim=-1)
    acc = ((predicted == labels).sum() / len(labels)).item()
    return acc


def mean_reciprocal_rank(probs: torch.Tensor, dim: int = -1) -> float:
    """MRR metric 계산 함수
    * NOTE: label이 0일 경우를 가정
    * probs (torch.Tensor): [b, # Classes]
    """
    ranks_raw = torch.argsort(probs, dim=dim, descending=True)
    _, ranks = torch.where(ranks_raw == 0)
    ranks = ranks + 1  # first rank is 1, not 0
    mrr = (1 / ranks).mean().item()
    return mrr


def topn_isin_topk(
    logits: torch.Tensor, n: int = 5, k: int = 5, return_as_tensor: bool = False
) -> Union[List[bool], torch.BoolTensor]:
    device = logits.get_device() if logits.get_device() != -1 else "cpu"

    _, indices = logits.float().topk(k=k, dim=-1)
    output = torch.zeros(indices.size(0), dtype=torch.long).to(device)

    for rank_idx in range(n):
        output += (rank_idx == indices).sum(dim=-1)

    output = output.bool()
    if return_as_tensor:
        return output
    return output.tolist()


def nPr(n: int, r: int, log_scale: bool = False):
    if log_scale:
        return np.sum([np.log(n - i) for i in range(r)])
    else:
        return int(np.round(np.exp(np.sum([np.log(n - i) for i in range(r)])), 1))


def random_prob_topM_topK(N: int, M: int = 1, K: int = 5):
    prob = 1 - np.exp(nPr(N - K, M, log_scale=True) - nPr(N, M, log_scale=True))
    return prob
