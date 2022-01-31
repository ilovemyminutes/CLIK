import torch
from torch import nn
from torchvision import models
import timm
from transformers import BertModel, BertConfig


class NonLinearMLP(nn.Module):
    """Non-linear multi layer perceptron. It consists of 2 fc layers and activation function between them.

    Args:
        input_dim (int):
        output_dim (int, optional): Defaults to 128.
        hidden_dim (int, optional): It will be the same as input_dim if it takes no value, Defaults to None.
        activation (str, optional): Name of activation function. Defaults to 'ReLU'.
            - NOTE. It should be the same as PyTorch activation function name. (ex. 'ReLU' for nn.ReLU())
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        hidden_dim: int = None,
        activation: str = "ReLU",
    ):
        super(NonLinearMLP, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.__dict__[activation]()
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.activation(x)
        x = self.linear_out(x)
        return x


class ImageEncoder(nn.Module):
    """Image Encoder used to train model with contrastive(or metric) learning.
    It consists of a raw image encoder and an mlp-based projector.

    Args:
        name (str): network name to load from torchvision
            - NOTE. for now it works fine with resnet-variants for torchvision.models
        output_dim (int, optional): last dim from mlp projector. Defaults to 128
        pretrained (bool, optional):
            - True: load network with pretrained weights
            - False: load just networkm not pretrained weights
    """

    def __init__(
        self, name: str = "resnet50", output_dim: int = 128, pretrained: bool = True
    ):
        super(ImageEncoder, self).__init__()

        # ResNet-variants: NOTE. it depends on torchvision (https://pytorch.org/vision/stable/index.html)
        if name.startswith("resnet"):
            self.enc = models.__dict__[name](pretrained=pretrained)
            enc_last_layer_input_dim = self.enc.fc.weight.shape[1]
            self.enc.fc = nn.Identity()

        # ViT-variants: NOTE. it depends on timm (https://github.com/rwightman/pytorch-image-models)
        elif name.startswith("vit"):  # vision transformer
            self.enc = timm.create_model(name, pretrained=pretrained)
            enc_last_layer_input_dim = self.enc.head.weight.shape[1]
            self.enc.head = nn.Identity()
        else:
            raise NotImplementedError

        self.proj = NonLinearMLP(enc_last_layer_input_dim, output_dim)

    def forward(self, x) -> torch.Tensor:
        x = self.enc(x)
        x = self.proj(x)
        return x


class TextEncoder(nn.Module):
    """Text Encoder used to train model with contrastive(or metric) learning.
    It consists of a raw text encoder and an mlp-based projector.

    Args:
        name (str): network name to load from transformers
            - NOTE. for now it works fine with transformers-bsed BERT models
        output_dim (int, optional): last dim from mlp projector. Defaults to 128
        pretrained (bool, optional):
            - True: load network with pretrained weights
            - False: load just networkm not pretrained weights
    """

    def __init__(
        self,
        name: str = "dsksd/bert-ko-small-minimal",
        output_dim: int = 128,
        pretrained: bool = True,
    ):
        super(TextEncoder, self).__init__()

        # BERT-variants: NOTE. it depends on transformers, HuggingFace (https://huggingface.co/docs/transformers/index)
        self.enc = (
            BertModel.from_pretrained(name)
            if pretrained
            else BertModel(BertConfig.from_pretrained(name))
        )
        enc_last_input_dim = self.enc.pooler.dense.weight.shape[1]
        self.proj = NonLinearMLP(enc_last_input_dim, output_dim)

    def forward(self, x) -> torch.Tensor:
        x = self.enc(**x)["pooler_output"]
        x = self.proj(x)
        return x
