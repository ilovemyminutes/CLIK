import os
import torch
from networks import CLIK
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from utils import Flags


def save_checkpoint(
    args: Flags,
    epoch: int,
    step: int,
    model: CLIK,
    optim_txt: Optimizer,
    optim_img: Optimizer,
    scheduler_txt: LambdaLR,
    scheduler_img: LambdaLR,
    save_path: str,
    is_distributed: bool = False,
):
    """체크포인트 저장 함수"""

    if is_distributed:
        ckpt = dict(
            epoch=epoch,
            step=step,
            model=model.module.state_dict(),
            optim_txt=optim_txt.state_dict(),
            optim_img=optim_img.state_dict(),
            scheduler_txt=scheduler_txt.state_dict(),
            scheduler_img=scheduler_img.state_dict(),
        )
        weights = dict(
            model=model.module.state_dict(),
            feature_dim=args.feature_dim,
            queue_size=args.queue_size,
            backbone_img=args.backbone_img,
            backbone_txt=args.backbone_txt,
            img_h=args.img_h,
            img_w=args.img_w,
            txt_max_length=args.txt_max_length,
        )

    else:
        ckpt = dict(
            epoch=epoch,
            step=step,
            model=model.state_dict(),
            optim_txt=optim_txt.state_dict(),
            optim_img=optim_img.state_dict(),
            scheduler_txt=scheduler_txt.state_dict(),
            scheduler_img=scheduler_img.state_dict(),
        )
        weights = dict(
            model=model.state_dict(),
            feature_dim=args.feature_dim,
            queue_size=args.queue_size,
            backbone_img=args.backbone_img,
            backbone_txt=args.backbone_txt,
            img_h=args.img_h,
            img_w=args.img_w,
            txt_max_length=args.txt_max_length,
        )

    save_path_weights = os.path.join(
        os.path.dirname(save_path), f"weights_{os.path.basename(save_path)}"
    )
    torch.save(ckpt, save_path)  # 체크포인트 저장
    torch.save(weights, save_path_weights)  # (추론에 활용되는) weight 파일 저장


def load_checkpoint(
    model: CLIK,
    optim_img: Optimizer,
    optim_txt: Optimizer,
    scheduler_img: LambdaLR,
    scheduler_txt: LambdaLR,
    ckpt_load_path: str,
    device: torch.device,
):
    """체크포인트 불러오기 함수. 학습을 이어가기 위해 시작 에폭/스텝을 함께 리턴"""
    ckpt = torch.load(ckpt_load_path, map_location=device)
    start_epoch = ckpt["epoch"] + 1
    start_step = ckpt["step"] + 1
    model.load_state_dict(ckpt["model"])  # model
    optim_txt.load_state_dict(ckpt["optim_txt"])  # optimizer
    optim_img.load_state_dict(ckpt["optim_img"])  # optimizer
    scheduler_txt.load_state_dict(ckpt["scheduler_txt"])  # scheduler
    scheduler_img.load_state_dict(ckpt["scheduler_img"])  # scheduler
    return (
        start_epoch,
        start_step,
        model,
        optim_txt,
        optim_img,
        scheduler_txt,
        scheduler_img,
    )
