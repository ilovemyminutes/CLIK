from abc import *
from typing import Dict, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .encoder import ImageEncoder, TextEncoder


class _CLIK(nn.Module, metaclass=ABCMeta):  # NOTE. made for further CLIK-variants
    def __init__(
        self,
        feature_dim: int,
        backbone_txt: str,
        backbone_img: str,
        pretrained: bool,
        temperature: float = 0.07,
    ):
        super(_CLIK, self).__init__()
        self.enc_context = TextEncoder(backbone_txt, feature_dim, pretrained)
        self.enc_instance = ImageEncoder(backbone_img, feature_dim, pretrained)
        self.temperature = temperature

    @abstractmethod
    def forward(
        self,
        matching: Dict[str, torch.Tensor],
        discrim: Dict[str, torch.Tensor],
        update_queue: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    @abstractmethod
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """편리한 추론을 위해 디자인 됨"""
        raise NotImplementedError

    @abstractmethod
    def get_semantic_matching_result(
        self, matching: Dict[str, torch.Tensor], update: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_preference_discrimination_result(
        self, discrim: Dict[str, torch.Tensor], *args
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _update_queue(self) -> None:
        raise NotImplementedError

    @property
    def device(self):
        d = next(self.parameters()).get_device()
        if d == -1:
            d = torch.device("cpu")
        return d

    def contrastive_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        loss = F.cross_entropy(logits / self.temperature, labels)
        return loss


class CLIK(_CLIK):
    def __init__(
        self,
        feature_dim: int,
        queue_size: int,
        backbone_txt: str = "dsksd/bert-ko-small-minimal",
        backbone_img: str = "vit_small_patch16_224_in21k",
        pretrained: bool = True,
        temperature: float = 0.07,
        rank: int = None,
    ):
        super(CLIK, self).__init__(
            feature_dim, backbone_txt, backbone_img, pretrained, temperature
        )
        # aggregation module
        self.agg = nn.Linear(2 * feature_dim, feature_dim)

        # instance queue
        self.register_buffer("inst_queue", torch.randn(queue_size, feature_dim))
        self.inst_queue = F.normalize(self.inst_queue)

        self.is_distributed = True if rank is not None else False
        self.rank = rank

    def forward(
        self, matching, discrim, update_queue: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            m_logits_cont_wise,
            m_logits_inst_wise,
            m_labels,
            m_loss,
        ) = self.get_semantic_matching_result(matching, update_queue)
        d_logits, d_labels, d_loss = self.get_preference_discrimination_result(discrim)
        return (
            m_logits_cont_wise,
            m_logits_inst_wise,
            m_labels,
            m_loss,
            d_logits,
            d_labels,
            d_loss,
        )

    @torch.no_grad()
    def predict(
        self, batch: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]
    ) -> torch.Tensor:
        logits, _ = self.get_preference_discrimination_result(batch, return_loss=False)
        return logits

    def get_semantic_matching_result(
        self,
        matching: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
        update_queue: bool = True,
        return_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Semantic Matching - understanding between shared contexts and instances

        Args:
            matching (Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]): batch for Semantic Matching
            update (bool, optional): [description]. Defaults to True.
            return_loss (bool, optional): [description]. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: [description]
        """
        contexts = F.normalize(
            self.enc_context(matching["contexts"])
        )  # [local B, feature_dim]
        instances = F.normalize(
            self.enc_instance(matching["instances"])
        )  # [local B, feature_dim]

        if self.is_distributed:
            contexts_gathered = [
                torch.zeros_like(contexts) for _ in range(dist.get_world_size())
            ]
            instances_gathered = [
                torch.zeros_like(instances) for _ in range(dist.get_world_size())
            ]

            dist.all_gather(tensor_list=contexts_gathered, tensor=contexts)
            dist.all_gather(tensor_list=instances_gathered, tensor=instances)

            contexts_gathered[self.rank] = contexts
            instances_gathered[self.rank] = instances

            contexts_gathered = torch.cat(contexts_gathered)  # [B, feature_dim]
            instances_gathered = torch.cat(instances_gathered)  # [B, feature_dim]

            # return 2 (local) logits for symmetric contrastive loss
            logits_contexts_wise = torch.mm(contexts, instances_gathered.T)
            logits_instances_wise = torch.mm(instances, contexts_gathered.T)

            labels = torch.arange(len(contexts_gathered), dtype=torch.long)[
                len(contexts) * self.rank : len(contexts) * (self.rank + 1)
            ].to(
                self.device
            )  # (local) pseudo-label

            assert logits_contexts_wise.size(0) == logits_instances_wise.size(0)
            assert labels.size(0) == logits_contexts_wise.size(0)

            if update_queue:
                self._update_queue(instances_gathered)

        else:
            logits_contexts_wise = torch.mm(
                contexts, instances.T
            )  # [matching_size, matching_size]
            logits_instances_wise = (
                logits_contexts_wise.T
            )  # [matching_size, matching_size]
            labels = torch.arange(len(logits_contexts_wise), dtype=torch.long).to(
                self.device
            )
            assert logits_contexts_wise.size(0) == logits_instances_wise.size(0)
            assert labels.size(0) == logits_contexts_wise.size(0)

            if update_queue:
                self._update_queue(instances)

        output = [logits_contexts_wise, logits_instances_wise, labels]
        if return_loss:
            output.append(
                (
                    self.contrastive_loss(logits_contexts_wise, labels)
                    + self.contrastive_loss(logits_instances_wise, labels)
                )
                / 2
            )
        return output

    def get_preference_discrimination_result(
        self,
        discrim: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
        return_loss: bool = True,
        return_energy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        D: # iterations of discrim
        K: # candidate instances
        C/H/W: channel/height/width of image
        """
        contexts = F.normalize(
            self.enc_context(discrim["contexts"])
        )  # [D, feature_dim]

        if discrim["instances"].ndim == 5:  # [D, K, C, H, W]
            logits, energy = [], []
            for context_row, instances_row in zip(
                contexts.unbind(dim=0), discrim["instances"].unbind(dim=0)
            ):
                context_row = context_row.unsqueeze(0)
                energy_row = F.softmax(torch.mm(context_row, self.inst_queue.T), dim=1)
                virt_instance = F.normalize(torch.mm(energy_row, self.inst_queue))
                joint_emb = F.normalize(
                    self.agg(torch.cat([context_row, virt_instance], dim=1))
                )

                instances_row = F.normalize(self.enc_instance(instances_row))

                logits.append(torch.mm(joint_emb, instances_row.T))
                energy.append(energy_row)

            logits = torch.vstack(logits)
            labels = torch.zeros(discrim["instances"].size(0), dtype=torch.long).to(
                self.device
            )  # [D]

        elif (
            discrim["instances"].ndim == 4
        ):  # instances: [K, C, H, W], contexts: [1, feature_dim]
            energy = F.softmax(torch.mm(contexts, self.inst_queue.T), dim=1)
            virt_instance = F.normalize(torch.mm(energy, self.inst_queue))
            joint_emb = F.normalize(
                self.agg(torch.cat([contexts, virt_instance], dim=1))
            )
            instances = F.normalize(self.enc_instance(discrim["instances"]))

            logits = torch.mm(joint_emb, instances.T)
            labels = torch.zeros(1, dtype=torch.long).to(self.device)

        output = [logits, labels]
        if return_loss:  # loss should be calculated in forward() during DDP
            output.append(self.contrastive_loss(logits, labels))
        if return_energy:  # NOTE. mainly for debugging
            if isinstance(energy, list):
                energy = torch.vstack(energy)
            output.append(energy)
        return output

    def _update_queue(self, instances: torch.Tensor) -> None:
        assert self.inst_queue.size(0) == instances.size(
            0
        ), f"For update, size should be the same between inst_queue({self.inst_queue.size(0)}) and instances({instances.size(0)})"
        self.inst_queue = instances.detach().float()


# NOTE Ablation - NO usage of shared context as anchor
class Ablation1(CLIK):
    def __init__(
        self,
        feature_dim: int,
        queue_size: int,
        backbone_txt: str = "dsksd/bert-ko-small-minimal",
        backbone_img: str = "vit_small_patch16_224_in21k",
        pretrained: bool = True,
        temperature: float = 0.07,
        rank: int = None,
    ):
        super(Ablation1, self).__init__(
            feature_dim, queue_size, backbone_txt, backbone_img, pretrained, temperature
        )

    def get_preference_discrimination_result(
        self,
        discrim: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
        return_loss: bool = True,
        return_energy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        D: # iterations of discrim
        K: # candidate instances
        C/H/W: channel/height/width of image
        """
        contexts = F.normalize(
            self.enc_context(discrim["contexts"])
        )  # [D, feature_dim]

        if discrim["instances"].ndim == 5:  # [D, K, C, H, W]
            logits, energy = [], []
            for context_row, instances_row in zip(
                contexts.unbind(dim=0), discrim["instances"].unbind(dim=0)
            ):
                # make virtual instance embeddings
                context_row = context_row.unsqueeze(0)
                energy_row = F.softmax(torch.mm(context_row, self.inst_queue.T), dim=1)
                virt_instance = F.normalize(torch.mm(energy_row, self.inst_queue))

                # get instance embeddings
                instances_row = F.normalize(self.enc_instance(instances_row))

                logits.append(torch.mm(virt_instance, instances_row.T))
                energy.append(energy_row)

            logits = torch.vstack(logits)
            labels = torch.zeros(discrim["instances"].size(0), dtype=torch.long).to(
                self.device
            )  # [D]

        elif (
            discrim["instances"].ndim == 4
        ):  # instances: [K, C, H, W], contexts: [1, feature_dim]
            # make virtual instance embeddings
            energy = F.softmax(torch.mm(contexts, self.inst_queue.T), dim=1)
            virt_instance = F.normalize(torch.mm(energy, self.inst_queue))

            # get instance embeddings
            instances = F.normalize(self.enc_instance(discrim["instances"]))

            logits = torch.mm(virt_instance, instances.T)
            labels = torch.zeros(1, dtype=torch.long).to(self.device)

        output = [logits, labels]
        if return_loss:  # loss should be calculated in forward() during DDP
            output.append(self.contrastive_loss(logits, labels))
        if return_energy:  # NOTE. mainly for debugging
            if isinstance(energy, list):
                energy = torch.vstack(energy)
            output.append(energy)
        return output

    def _update_queue(self, instances: torch.Tensor) -> None:
        assert self.inst_queue.size(0) == instances.size(
            0
        ), f"For update, size should be the same between inst_queue({self.inst_queue.size(0)}) and instances({instances.size(0)})"
        self.inst_queue = instances.detach().float()
