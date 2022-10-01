from abc import *
from typing import Tuple, Dict, Union, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .encoder import ImageEncoder, TextEncoder

MatchingBatch = Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]
RankingBatch = Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]

MatchingOutput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
MatchingOutputWithLoss = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
RankingOutput = Tuple[torch.Tensor, torch.Tensor]
RankingOutputWithLoss = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class _CLIK(nn.Module, metaclass=ABCMeta):
    """
    An abstract class for CLIK that is made for further CLIK-variation
    """
    def __init__(
            self,
            feature_dim: int,
            backbone_txt: str,
            backbone_img: str,
            memory_bank_size: int,
            pretrained: bool,
            temperature: float = 0.07,
    ):
        super(_CLIK, self).__init__()
        # Dual-encoder
        self.txt_encoder = TextEncoder(backbone_txt, feature_dim, pretrained)
        self.img_encoder = ImageEncoder(backbone_img, feature_dim, pretrained)

        # Aggregation Module (fully connected layer)
        self.agg = nn.Linear(2 * feature_dim, feature_dim)

        # Memory Bank
        self.register_buffer('memory_bank', torch.randn(memory_bank_size, feature_dim))
        self.memory_bank = F.normalize(self.memory_bank)
        self.temperature = temperature

    @abstractmethod
    def forward(
            self,
            matching_batch: Dict[str, torch.Tensor],
            ranking_batch: Dict[str, torch.Tensor],
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
            self, matching_batch: Dict[str, torch.Tensor], update_bank: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_image_ranking_result(
            self, ranking_batch: Dict[str, torch.Tensor], *args
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _update_bank(self, images: torch.Tensor) -> None:
        if self.memory_bank.size(0) != images.size(0):
            raise ValueError(f'For update, the size of images ({images.size(0)}) should be the same as '
                             f'that of memory_bank ({self.memory_bank.size(0)}).')
        self.memory_bank: torch.Tensor = images.detach().float()

    def contrastive_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        loss = F.cross_entropy(logits / self.temperature, labels)
        return loss

    @property
    def device(self):
        d = next(self.parameters()).get_device()
        if d == -1:
            d = torch.device("cpu")
        return d


class CLIK(_CLIK):
    def __init__(
            self,
            feature_dim: int,
            memory_bank_size: int,
            backbone_txt: str = 'dsksd/bert-ko-small-minimal',
            backbone_img: str = 'vit_small_patch16_224_in21k',
            pretrained: bool = True,
            temperature: float = 0.07,
            rank: int = None,
    ):
        super(CLIK, self).__init__(feature_dim, backbone_txt, backbone_img, memory_bank_size, pretrained, temperature)
        self.rank = rank
        self.is_distributed = True if rank is not None else False

    def forward(
            self,
            matching_batch,
            ranking_batch,
            update_bank: bool = True
    ) -> Tuple[MatchingOutputWithLoss, RankingOutputWithLoss]:
        matching_result: MatchingOutputWithLoss = self.get_topic_matching_result(matching_batch, update_bank)
        ranking_result: RankingOutputWithLoss = self.get_image_ranking_result(ranking_batch)
        return matching_result, ranking_result

    @torch.no_grad()
    def predict(self, ranking_batch: RankingBatch) -> torch.Tensor:
        compatibility_scores, _ = self.get_image_ranking_result(ranking_batch, return_loss=False)
        return compatibility_scores

    def get_topic_matching_result(
            self,
            matching_batch: MatchingBatch,
            update_bank: bool = True,
            return_loss: bool = True,
    ) -> Union[MatchingOutput, MatchingOutputWithLoss]:
        """Topic Matching: understanding between topics and images

        Args:
            matching_batch (MatchingBatch): batch for Semantic Matching
            update_bank (bool, optional): [description]. Defaults to True.
            return_loss (bool, optional): [description]. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: [description]

        Notations for explaination
        * local B: batch_size for each process (if you use single gpu, it is equal to total batch_size)
        * B: total batch_size (if you use single gpu, it is equal to local B)
        * h: feature_dim
        """
        topics: torch.Tensor = F.normalize(self.txt_encoder(matching_batch['topics']))  # [local B, h]
        images: torch.Tensor = F.normalize(self.img_encoder(matching_batch['images']))  # [local B, h]

        if self.is_distributed:  # distributed training
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
            topics_gathered: torch.Tensor = torch.cat(topics_gathered)  # [B, h]
            images_gathered: torch.Tensor = torch.cat(images_gathered)  # [B, h]

            # logits & labels for Loss_{matching}
            logits_topic_wise: torch.Tensor = torch.mm(topics, images_gathered.T)  # [local B, B]
            logits_image_wise: torch.Tensor = torch.mm(images, topics_gathered.T)  # [local B, B]
            labels: torch.Tensor = (torch.arange(len(topics_gathered), dtype=torch.long)
                                    [len(topics) * self.rank: len(topics) * (self.rank + 1)]
                                    .to(self.device))  # [local B]
            if update_bank:
                self._update_bank(images_gathered)
        else:  # single-gpu training
            # logits & labels for Loss_{matching}
            logits_topic_wise: torch.Tensor = torch.mm(topics, images.T)  # [local B, local B]
            logits_image_wise: torch.Tensor = logits_topic_wise.T  # [local B, local B]
            labels = torch.arange(len(logits_topic_wise), dtype=torch.long).to(self.device)  # [local B]
            if update_bank:
                self._update_bank(images)

        if return_loss:
            loss_s2i: torch.Tensor = self.contrastive_loss(logits_topic_wise, labels)
            loss_i2s: torch.Tensor = self.contrastive_loss(logits_image_wise, labels)
            loss_matching: torch.Tensor = (loss_s2i + loss_i2s) / 2
            output: MatchingOutputWithLoss = (logits_topic_wise, logits_image_wise, labels, loss_matching)
        else:
            output: MatchingOutput = (logits_topic_wise, logits_image_wise, labels)
        return output

    def get_image_ranking_result(
            self, ranking_batch: RankingBatch, return_loss: bool = True
    ) -> Union[RankingOutput, RankingOutputWithLoss]:
        """
        G: # sampled groups
        K: # sampled images for each group
        M: size of Memory Bank
        C/H/W: channel/height/width of image
        h: feature_dim
        """
        if ranking_batch['images'].ndim != 5 and ranking_batch['images'].ndim != 4:
            raise NotImplementedError(f"ranking_batch['images'] has wrong dimension: {ranking_batch['images'].ndim}")

        topics = F.normalize(self.txt_encoder(ranking_batch['topics']))  # [G, h]

        if ranking_batch['images'].ndim == 5:  # G > 1 (batch['images']: [G, K, C, H, W])
            num_groups: int = ranking_batch['images'].size(0)
            compatibility_scores: List[torch.Tensor] = []
            for topic, images in zip(topics.unbind(dim=0), ranking_batch['images'].unbind(dim=0)):
                topic = topic.unsqueeze(0)  # [1, h]
                images = F.normalize(self.img_encoder(images))  # [K, h]
                group_query: torch.Tensor = self.generate_group_query(topic)
                compatibility_scores.append(torch.mm(group_query, images.T))  # append [1, K]
            compatibility_scores: torch.Tensor = torch.vstack(compatibility_scores)
        else:  # G == 1 (batch['images']: [K, C, H, W])
            num_groups = 1
            images: torch.Tensor = F.normalize(self.img_encoder(ranking_batch['images']))  # [K, h]
            group_query: torch.Tensor = self.generate_group_query(topics)  # [1, h]
            compatibility_scores: torch.Tensor = torch.mm(group_query, images.T)  # [1, K]

        # logits & labels for Loss_{ranking}
        logits: torch.Tensor = compatibility_scores
        labels: torch.Tensor = torch.zeros(num_groups, dtype=torch.long).to(self.device)

        if return_loss:
            loss_ranking: torch.Tensor = self.contrastive_loss(logits, labels)
            output: RankingOutputWithLoss = (logits, labels, loss_ranking)
        else:
            output: RankingOutput = (logits, labels)
        return output

    def generate_group_query(self, topic_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            topic_embedding: [1, feature_dim]
        Returns:
            group_query: [1, feature_dim]

        """
        energy = F.softmax(torch.mm(topic_embedding, self.memory_bank.T), dim=1)  # [1, M]
        virtual_img = F.normalize(torch.mm(energy, self.memory_bank))  # [1, h]
        group_query = F.normalize(self.agg(torch.cat([topic_embedding, virtual_img], dim=1)))  # [1, h]
        return group_query


