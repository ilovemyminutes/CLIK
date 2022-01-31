import os
from typing import List, Tuple
import torch
import torch.distributed as dist
from torch.utils.data import Sampler, DistributedSampler, BatchSampler
from .metric import topn_isin_topk, accuracy, mean_reciprocal_rank


def setup(rank, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def aggregate_data(*data) -> List[torch.Tensor]:
    aggregated = []
    for d in data:
        gathered = [
            torch.zeros_like(d, dtype=d.dtype) for _ in range(dist.get_world_size())
        ]
        dist.barrier()
        dist.all_gather(tensor_list=gathered, tensor=d)
        aggregated.append(torch.cat(gathered))
    return aggregated


class DistributedSampler(Sampler):
    """Iterable wrapper that distributes data across multiple workers.
    Args:
        iterable (iterable)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within ``num_replicas``.
    Example:
        >>> list(DistributedSampler(range(10), num_replicas=2, rank=0))
        [0, 2, 4, 6, 8]
        >>> list(DistributedSampler(range(10), num_replicas=2, rank=1))
        [1, 3, 5, 7, 9]

    Reference: https://github.com/PetrochukM/PyTorch-NLP
    """

    def __init__(self, iterable, num_replicas=None, rank=None):
        self.iterable = iterable
        self.num_replicas = num_replicas
        self.rank = rank

        if num_replicas is None or rank is None:  # pragma: no cover
            if not dist.is_initialized():
                raise RuntimeError("Requires `torch.distributed` to be initialized.")

            self.num_replicas = (
                dist.get_world_size() if num_replicas is None else num_replicas
            )
            self.rank = dist.get_rank() if rank is None else rank

        if self.rank >= self.num_replicas:
            raise IndexError("`rank` must be smaller than the `num_replicas`.")

    def __iter__(self):
        return iter(
            [
                e
                for i, e in enumerate(self.iterable)
                if (i - self.rank) % self.num_replicas == 0
            ]
        )

    def __len__(self):
        return len(self.iterable)


class DistributedBatchSampler(BatchSampler):
    """`BatchSampler` wrapper that distributes across each batch multiple workers.
    Args:
        batch_sampler (torch.utils.data.sampler.BatchSampler)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within num_replicas.
    Example:
        >>> from torch.utils.data.sampler import BatchSampler
        >>> from torch.utils.data.sampler import SequentialSampler
        >>> sampler = SequentialSampler(list(range(12)))
        >>> batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=False)
        >>>
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=0))
        [[0, 2], [4, 6], [8, 10]]
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=1))
        [[1, 3], [5, 7], [9, 11]]

    Reference: https://github.com/PetrochukM/PyTorch-NLP
    """

    def __init__(self, batch_sampler, num_replicas: int, rank: int):
        self.batch_sampler = batch_sampler
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        for batch in self.batch_sampler:
            yield list(
                DistributedSampler(
                    batch, num_replicas=self.num_replicas, rank=self.rank
                )
            )

    def __len__(self):
        return len(self.batch_sampler)


def step_log_for_dist_training(
    loss: torch.Tensor,
    m_loss: torch.Tensor,
    d_loss: torch.Tensor,
    m_logits_cont_wise: torch.Tensor,
    m_logits_inst_wise: torch.Tensor,
    m_labels: torch.Tensor,
    d_logits: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    loss_dist = torch.tensor([loss.item()], dtype=torch.float).to(device)
    m_loss_dist = torch.tensor([m_loss.item()], dtype=torch.float).to(device)
    d_loss_dist = torch.tensor([d_loss.item()], dtype=torch.float).to(device)
    m_acc_cont_wise = torch.tensor(
        [accuracy(m_logits_cont_wise, m_labels)], dtype=torch.float
    ).to(device)
    m_acc_inst_wise = torch.tensor(
        [accuracy(m_logits_inst_wise, m_labels)], dtype=torch.float
    ).to(device)
    d_mrr = torch.tensor([mean_reciprocal_rank(d_logits)], dtype=torch.float).to(device)
    d_top1_top1_acc = topn_isin_topk(d_logits, n=1, k=1, return_as_tensor=True)
    d_top3_top1_acc = topn_isin_topk(d_logits, n=3, k=1, return_as_tensor=True)
    d_top5_top1_acc = topn_isin_topk(d_logits, n=5, k=1, return_as_tensor=True)
    d_top5_top5_acc = topn_isin_topk(d_logits, n=5, k=5, return_as_tensor=True)

    (
        loss_dist,
        m_loss_dist,
        d_loss_dist,
        m_acc_cont_wise,
        m_acc_inst_wise,
        d_mrr,
        d_top1_top1_acc,
        d_top3_top1_acc,
        d_top5_top1_acc,
        d_top5_top5_acc,
    ) = aggregate_data(
        loss_dist,
        m_loss_dist,
        d_loss_dist,
        m_acc_cont_wise,
        m_acc_inst_wise,
        d_mrr,
        d_top1_top1_acc,
        d_top3_top1_acc,
        d_top5_top1_acc,
        d_top5_top5_acc,
    )
    loss_dist = loss_dist.mean().item()
    m_loss_dist = m_loss_dist.mean().item()
    d_loss_dist = d_loss_dist.mean().item()
    m_acc_cont_wise = m_acc_cont_wise.mean().item()
    m_acc_inst_wise = m_acc_inst_wise.mean().item()
    d_mrr = d_mrr.mean().item()
    d_top1_top1_acc = d_top1_top1_acc.float().mean().item()
    d_top3_top1_acc = d_top3_top1_acc.float().mean().item()
    d_top5_top1_acc = d_top5_top1_acc.float().mean().item()
    d_top5_top5_acc = d_top5_top5_acc.float().mean().item()

    return (
        loss_dist,
        m_loss_dist,
        d_loss_dist,
        m_acc_cont_wise,
        m_acc_inst_wise,
        d_mrr,
        d_top1_top1_acc,
        d_top3_top1_acc,
        d_top5_top1_acc,
        d_top5_top5_acc,
    )
