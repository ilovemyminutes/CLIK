import os
import warnings
import argparse
import torch
from inference import TaskOperator
from preprocessing import TextPreprocessor, get_eval_transforms
from utils import Flags
from networks import CLIK


def demo_infer(args):
    device = torch.device("cpu")
    config = Flags(args.config_path).get()

    # load model
    model = CLIK(
        feature_dim=config.feature_dim, queue_size=config.queue_size, pretrained=False
    )
    if config.ckpt_load_path is not None:
        assert os.path.isfile(
            config.ckpt_load_path
        ), f"There's no file '{config.ckpt_load_path}'"
        ckpt = torch.load(args.ckpt_load_path, map_location="cpu")
        if "model" in ckpt.keys():
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
    else:
        warnings.warn(
            "A model will be initialized randomly since 'ckpt_load_path' was not set."
        )

    # load preprocessors
    txt_preprocessor = TextPreprocessor(
        pretrained_tokenizer=config.backbone_txt, dropout=0.0
    )
    img_transforms = get_eval_transforms(config.img_h, config.img_w)

    # define inference operator
    operator = TaskOperator(
        model=model,
        img_transforms=img_transforms,
        txt_preprocessor=txt_preprocessor,
        device=device,
    )

    # example
    plan_attrs = dict(
        name="삼성 블루투스 스마트 키보드 트리오 500 무선 멀티페어링 미니 텐키리스",
        cat1="멘즈",
        cat2="디지털",
        kwds="블루투스키보드,삼성키보드,텐키리스키보드",
    )

    prod_paths = [
        "http://shop1.phinf.naver.net/20210906_275/1630909046510r69qH_JPEG/32044944987775788_520691521.jpg",
        "http://shop1.phinf.naver.net/20210524_26/1621822172189K8buH_JPEG/22958017887527000_1622201454.jpg",
        "http://shop1.phinf.naver.net/20210524_266/1621822172391CLdmp_JPEG/22958018094581723_210217066.jpg",
        "http://shop1.phinf.naver.net/20210524_122/1621822172592nBy3X_JPEG/22958018296120841_1391937211.jpg",
        "http://shop1.phinf.naver.net/20210524_52/1621822173009CSriT_JPEG/22958018497795089_1002902940.jpg",
    ]

    print(
        "[+] Plan Attributes\n",
        f"Name: {plan_attrs['name']}\n",
        f"Category 1: {plan_attrs['cat1']}\n",
        f"Category 2: {plan_attrs['cat2']}\n",
        f"Keywords: {plan_attrs['kwds']}\n",
    )

    best_urls = operator(plan_attrs, prod_paths, topk=args.topk)
    print("[+] Best URLs")
    for i, url in enumerate(best_urls):
        print(f"Top{i+1}: {url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="./configs/CLIK.yaml")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()
    assert os.path.isfile(args.config_path), f"There's no file '{args.config_path}'"

    demo_infer(args)
