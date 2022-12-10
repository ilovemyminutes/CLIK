import argparse
import warnings

import torch

from inference import Predictor
from networks import CLIK
from preprocessing import TextPreprocessor, get_eval_transforms
from utils import Flags


def demo_predict(args):
    device = torch.device("cpu")
    config = Flags(args.config_path).get()

    # load model
    model = CLIK(
        feature_dim=config.feature_dim,
        memory_bank_size=config.memory_bank_size,
        pretrained=False,
    )
    if config.ckpt_load_path is not None:
        ckpt = torch.load(args.ckpt_load_path, map_location="cpu")
        if "model" in ckpt.keys():
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
    else:
        warnings.warn(
            "A model will be initialized randomly since 'ckpt_load_path' was not set."
        )

    # initialize predictor
    txt_preprocessor = TextPreprocessor(
        pretrained_tokenizer=config.backbone_txt, dropout=0.0
    )
    img_transforms = get_eval_transforms(config.img_h, config.img_w)
    predictor = Predictor(
        model=model,
        img_transforms=img_transforms,
        txt_preprocessor=txt_preprocessor,
        device=device,
    )

    # topic & image information
    topic_attrs = dict(
        name="Bluetooth smart keyboard multi-pairing mini tenkeyless keyboard",
        cat1="mens",
        cat2="digital",
        kwds="BluetoothKeyboard,Samsung,TenkeylessKeyboard",
    )
    prod_paths = [
        "[IMAGE_URL1]",
        "[IMAGE_URL2]",
        "[IMAGE_URL3]",
        "[IMAGE_URL4]",
        "[IMAGE_URL5]",
    ]
    print(
        "[+] Topic Attributes\n",
        f"Name: {topic_attrs['name']}\n",
        f"Category 1: {topic_attrs['cat1']}\n",
        f"Category 2: {topic_attrs['cat2']}\n",
        f"Keywords: {topic_attrs['kwds']}\n",
    )

    # predict
    best_urls = predictor(topic_attrs, prod_paths, topk=args.topk)
    print("[+] Best URLs")
    for i, url in enumerate(best_urls):
        print(f"Top{i + 1}: {url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="./configs/CLIK.yaml")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()
    demo_predict(args)
