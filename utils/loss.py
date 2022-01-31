import torch
from torch import nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2, reduction: str = "sum"):
        super(TripletLoss, self).__init__()
        self.m = margin
        self.reduction = reduction

    def forward(
        self, a: torch.Tensor, p: torch.Tensor, n: torch.Tensor
    ) -> torch.Tensor:
        d_pos = F.pairwise_distance(a, p, p=2.0)
        d_neg = F.pairwise_distance(a, n, p=2.0)
        loss = F.relu(d_pos - d_neg + self.m)
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        else:
            raise NotImplementedError
        return loss


class PairwiseRankingLoss(nn.Module):
    """Pairwise Ranking Loss
    Reference. Zhichen Zhao et. al, 2019, What You Look Matters? Offline Evaluation of Advertising Creatives for Cold-start Problem
    """

    def __init__(self, reduction="mean"):
        super(PairwiseRankingLoss, self).__init__()
        self.bceloss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, s_pos: torch.Tensor, s_neg: torch.Tensor):
        device = (
            torch.device(s_pos.get_device())
            if s_pos.get_device() != -1
            else torch.device("cpu")
        )
        diff = (s_pos - s_neg).squeeze()
        labels = torch.ones(s_pos.size(0), dtype=torch.float).to(device)
        loss = self.bceloss(diff, labels)
        return loss
