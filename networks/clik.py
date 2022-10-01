from abc import *
from typing import Tuple, Dict, Union
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
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
        self.txt_encoder = TextEncoder(backbone_txt, feature_dim, pretrained)
        self.img_encoder = ImageEncoder(backbone_img, feature_dim, pretrained)
        self.temperature = temperature

    @abstractmethod
    def forward(
        self,
        matching: Dict[str, torch.Tensor],
        discrim: Dict[str, torch.Tensor],
        update_bank: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    @abstractmethod
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """편리한 추론을 위해 디자인 됨"""
        raise NotImplementedError

    @abstractmethod
    def get_topic_matching_result(
        self, matching: Dict[str, torch.Tensor], update: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_image_ranking_result(
        self, discrim: Dict[str, torch.Tensor], *args
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _update_bank(self) -> None:
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

        # memory bank
        self.register_buffer("memory_bank", torch.randn(queue_size, feature_dim))
        self.memory_bank = F.normalize(self.memory_bank)

        self.is_distributed = True if rank is not None else False
        self.rank = rank

    def forward(
        self, matching, discrim, update_bank: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            m_logits_cont_wise,
            m_logits_inst_wise,
            m_labels,
            m_loss,
        ) = self.get_topic_matching_result(matching, update_bank)
        d_logits, d_labels, d_loss = self.get_image_ranking_result(discrim)
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
        logits, _ = self.get_image_ranking_result(batch, return_loss=False)
        return logits

    def get_topic_matching_result(
        self,
        matching: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
        update_bank: bool = True,
        return_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Semantic Matching - understanding between shared contexts and instances

        Args:
            matching (Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]): batch for Semantic Matching
            update_bank (bool, optional): [description]. Defaults to True.
            return_loss (bool, optional): [description]. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: [description]
        """
        contexts = F.normalize(
            self.txt_encoder(matching["contexts"])
        )  # [local B, feature_dim]
        instances = F.normalize(
            self.img_encoder(matching["instances"])
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

            if update_bank:
                self._update_bank(instances_gathered)

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

            if update_bank:
                self._update_bank(instances)

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

    def get_image_ranking_result(
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
            self.txt_encoder(discrim["contexts"])
        )  # [D, feature_dim]

        if discrim["instances"].ndim == 5:  # [D, K, C, H, W]
            logits, energy = [], []
            for context_row, instances_row in zip(
                contexts.unbind(dim=0), discrim["instances"].unbind(dim=0)
            ):
                context_row = context_row.unsqueeze(0)
                energy_row = F.softmax(torch.mm(context_row, self.memory_bank.T), dim=1)
                virt_instance = F.normalize(torch.mm(energy_row, self.memory_bank))
                joint_emb = F.normalize(
                    self.agg(torch.cat([context_row, virt_instance], dim=1))
                )

                instances_row = F.normalize(self.img_encoder(instances_row))

                logits.append(torch.mm(joint_emb, instances_row.T))
                energy.append(energy_row)

            logits = torch.vstack(logits)
            labels = torch.zeros(discrim["instances"].size(0), dtype=torch.long).to(
                self.device
            )  # [D]

        elif (
            discrim["instances"].ndim == 4
        ):  # instances: [K, C, H, W], contexts: [1, feature_dim]
            energy = F.softmax(torch.mm(contexts, self.memory_bank.T), dim=1)
            virt_instance = F.normalize(torch.mm(energy, self.memory_bank))
            joint_emb = F.normalize(
                self.agg(torch.cat([contexts, virt_instance], dim=1))
            )
            instances = F.normalize(self.img_encoder(discrim["instances"]))

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

    def _update_bank(self, instances: torch.Tensor) -> None:
        assert self.memory_bank.size(0) == instances.size(
            0
        ), f"For update, size should be the same between memory_bank({self.memory_bank.size(0)}) and instances({instances.size(0)})"
        self.memory_bank = instances.detach().float()
