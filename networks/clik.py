from abc import *
from typing import Tuple, Dict, Union, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .encoder import ImageEncoder, TextEncoder

MatchingOutput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
MatchingOutputWithLoss = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


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
            memory_bank_size: int,
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
        self.register_buffer("memory_bank", torch.randn(memory_bank_size, feature_dim))
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
    ) -> Union[MatchingOutput, MatchingOutputWithLoss]:
        """Topic Matching: understanding between topics and images

        Args:
            matching (Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]): batch for Semantic Matching
            update_bank (bool, optional): [description]. Defaults to True.
            return_loss (bool, optional): [description]. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: [description]

        Notations for explaination
        * local B: batch_size for each process (if you use single gpu, it is equal to total batch_size)
        * B: total batch_size (if you use single gpu, it is equal to local B)
        * H: feature_dim
        """
        topics: torch.Tensor = F.normalize(self.txt_encoder(matching["contexts"]))  # [local B, H]
        images: torch.Tensor = F.normalize(self.img_encoder(matching["instances"]))  # [local B, H]

        # distributed training
        if self.is_distributed:
            # gather data from multiple processes
            topics_gathered: List[torch.Tensor] = [
                torch.zeros_like(topics) for _ in range(dist.get_world_size())
            ]
            images_gathered: List[torch.Tensor] = [
                torch.zeros_like(images) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(tensor_list=topics_gathered, tensor=topics)
            dist.all_gather(tensor_list=images_gathered, tensor=images)
            topics_gathered[self.rank]: torch.Tensor = topics
            images_gathered[self.rank]: torch.Tensor = images
            topics_gathered: torch.Tensor = torch.cat(topics_gathered)  # [B, H]
            images_gathered: torch.Tensor = torch.cat(images_gathered)  # [B, H]

            # logits & labels for Loss_{matching}
            logits_topic_wise: torch.Tensor = torch.mm(topics, images_gathered.T)  # [local B, B]
            logits_image_wise: torch.Tensor = torch.mm(images, topics_gathered.T)  # [local B, B]
            labels: torch.Tensor = (torch.arange(len(topics_gathered), dtype=torch.long)
                                    [len(topics) * self.rank: len(topics) * (self.rank + 1)]
                                    .to(self.device))  # [local B]

            # update Memory Bank
            if update_bank:
                self._update_bank(images_gathered)

        # single-gpu training
        else:
            # logits & labels for Loss_{matching}
            logits_topic_wise: torch.Tensor = torch.mm(topics, images.T)  # [local B, local B]
            logits_image_wise: torch.Tensor = logits_topic_wise.T  # [local B, local B]
            labels = torch.arange(len(logits_topic_wise), dtype=torch.long).to(self.device)  # [local B]

            # update Memory Bank
            if update_bank:
                self._update_bank(images)

        # return output
        if return_loss:
            loss_s2i: torch.Tensor = self.contrastive_loss(logits_topic_wise, labels)
            loss_i2s: torch.Tensor = self.contrastive_loss(logits_image_wise, labels)
            loss_clik: torch.Tensor = (loss_s2i + loss_i2s) / 2
            output: MatchingOutputWithLoss = (logits_topic_wise, logits_image_wise, labels, loss_clik)
        else:
            output: MatchingOutput = (logits_topic_wise, logits_image_wise, labels)
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
