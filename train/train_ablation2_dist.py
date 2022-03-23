import os
from typing import Dict, Union, Tuple, List
import warnings
from tqdm import tqdm
import torch
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from networks.clik import Ablation2
from preprocessing import (
    get_train_transforms,
    get_eval_transforms,
    TextPreprocessor,
)
from utils.ddp_utils import setup, aggregate_data, step_log_for_dist_training
from utils.data_utils import compose_dataloaders_ablation2, load_meta_data, txt_input_to_device
from utils.flags import Flags
from utils.metric import accuracy, mean_reciprocal_rank, topn_isin_topk
from utils.logger import Logger
from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils import print_gpu_status, set_seed



def compose_batch(
    contexts: Dict[str, torch.Tensor],
    instances: torch.Tensor,
    device: torch.device,
    scores: List[torch.Tensor] = None
) -> Tuple[
    Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
    Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
]:
    contexts, instances = txt_input_to_device(contexts, device), instances.to(
        device
    )  # align data to device
    batch = dict(contexts=contexts, instances=instances)

    if scores is not None:
        scores = torch.vstack(scores).T.to(device)
        batch['scores'] = scores

    return batch


def train_one_epoch(
    model: Ablation2,
    m_loader: DataLoader,
    d_loader: DataLoader,
    optim_txt: Optimizer,
    optim_img: Optimizer,
    scheduler_txt: LambdaLR,
    scheduler_img: LambdaLR,
    cur_epoch: int,
    tot_epoch: int,
    scaler: GradScaler,
    device: torch.device,
    logger: Logger = None,
):
    model.train()
    for step, ((m_plc, m_pri), (d_plc, d_pri, d_scores)) in tqdm(
        enumerate(zip(m_loader, d_loader)), desc=f"[Train: {cur_epoch}/{tot_epoch-1}]"
    ):
        matching = compose_batch(m_plc, m_pri, device=device)
        discrim = compose_batch(d_plc, d_pri, device=device, scores=d_scores)

        with autocast(enabled=True):
            (
                m_logits_cont_wise,
                m_logits_inst_wise,
                m_labels,
                m_loss,
                d_logits,
                _,
                d_loss,
            ) = model(matching, discrim)
            loss = 20 * m_loss + d_loss

        optim_txt.zero_grad()
        optim_img.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optim_txt)
        scaler.step(optim_img)
        scaler.update()
        scheduler_txt.step()
        scheduler_img.step()

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
        ) = step_log_for_dist_training(
            loss,
            m_loss,
            d_loss,
            m_logits_cont_wise,
            m_logits_inst_wise,
            m_labels,
            d_logits,
            device,
        )  # logging for distributed training

        if device.index == 0:  # rank = 0
            train_log = dict(
                epoch=cur_epoch,
                step=step + (cur_epoch * len(m_loader)),
                lr_txt=scheduler_txt.get_last_lr()[0],
                lr_img=scheduler_img.get_last_lr()[0],
                train_loss=loss_dist,
                train_m_loss=m_loss_dist,
                train_d_loss=d_loss_dist,
                train_m_acc_cont_wise=m_acc_cont_wise,
                train_m_acc_inst_wise=m_acc_inst_wise,
                train_mrr=d_mrr,
                train_top1top1_acc=d_top1_top1_acc,
                train_top3top1_acc=d_top3_top1_acc,
                train_top5top1_acc=d_top5_top1_acc,
                train_top5top5_acc=d_top5_top5_acc,
            )

            logger.record(train_log)

        dist.barrier()


def valid_one_epoch(
    model: Ablation2,
    m_loader: DataLoader,
    d_loader: DataLoader,
    cur_epoch: int,
    tot_epoch: int,
    device: torch.device,
):
    model.eval()

    losses = []  # overall loss
    m_losses = []  # loss of S.M.
    d_losses = []  # loss of P.D.
    inbatch_m_accs_cont_wise = []  # S.M accuracy (context-wise)
    inbatch_m_accs_inst_wise = []  # S.M accuracy (instance-wise)
    d_mrrs = []  # P.D. MRR
    d_top1_top1_accs = []  # P.D. Top1-Top1 accuracy
    d_top3_top1_accs = []  # P.D. Top3-Top1 accuracy
    d_top5_top1_accs = []  # P.D. Top5-Top1 accuracy
    d_top5_top5_accs = []  # P.D. Top5-Top5 accuracy

    for (m_plc, m_pri), (d_plc, d_pri) in tqdm(
        zip(m_loader, d_loader), desc=f"[Valid: {cur_epoch}/{tot_epoch-1}]"
    ):
        matching = compose_batch(m_plc, m_pri, device=device)
        discrim = compose_batch(d_plc, d_pri, device=device)
        with autocast(enabled=True):
            with torch.no_grad():
                (
                    m_logits_cont_wise,
                    m_logits_inst_wise,
                    m_labels,
                    m_loss,
                    d_logits,
                    _,
                    d_loss,
                ) = model(matching, discrim, update_queue=False)

            loss = 20 * m_loss + d_loss

        m_acc_cont_wise = accuracy(m_logits_cont_wise, m_labels)
        m_acc_inst_wise = accuracy(m_logits_inst_wise, m_labels)
        discrim_mrr = mean_reciprocal_rank(d_logits)
        discrim_top1_top1_acc = topn_isin_topk(d_logits, n=1, k=1)
        discrim_top3_top1_acc = topn_isin_topk(d_logits, n=3, k=1)
        discrim_top5_top1_acc = topn_isin_topk(d_logits, n=5, k=1)
        discrim_top5_top5_acc = topn_isin_topk(d_logits, n=5, k=5)

        losses.append(loss.item())
        m_losses.append(m_loss.item())
        d_losses.append(d_loss.item())
        inbatch_m_accs_cont_wise.append(m_acc_cont_wise)
        inbatch_m_accs_inst_wise.append(m_acc_inst_wise)
        d_mrrs.append(discrim_mrr)
        d_top1_top1_accs.extend(discrim_top1_top1_acc)
        d_top3_top1_accs.extend(discrim_top3_top1_acc)
        d_top5_top1_accs.extend(discrim_top5_top1_acc)
        d_top5_top5_accs.extend(discrim_top5_top5_acc)

    losses = torch.tensor(losses, dtype=torch.float).to(device)
    m_losses = torch.tensor(m_losses, dtype=torch.float).to(device)
    d_losses = torch.tensor(d_losses, dtype=torch.float).to(device)
    inbatch_m_accs_cont_wise = torch.tensor(
        inbatch_m_accs_cont_wise, dtype=torch.float
    ).to(device)
    inbatch_m_accs_inst_wise = torch.tensor(
        inbatch_m_accs_inst_wise, dtype=torch.float
    ).to(device)
    d_mrrs = torch.tensor(d_mrrs, dtype=torch.float).to(device)
    d_top1_top1_accs = torch.tensor(d_top1_top1_accs, dtype=torch.bool).to(device)
    d_top3_top1_accs = torch.tensor(d_top3_top1_accs, dtype=torch.bool).to(device)
    d_top5_top1_accs = torch.tensor(d_top5_top1_accs, dtype=torch.bool).to(device)
    d_top5_top5_accs = torch.tensor(d_top5_top5_accs, dtype=torch.bool).to(device)

    dist.barrier()

    (
        losses,
        m_losses,
        d_losses,
        inbatch_m_accs_cont_wise,
        inbatch_m_accs_inst_wise,
        d_mrrs,
        d_top1_top1_accs,
        d_top3_top1_accs,
        d_top5_top1_accs,
        d_top5_top5_accs,
    ) = aggregate_data(
        losses,
        m_losses,
        d_losses,
        inbatch_m_accs_cont_wise,
        inbatch_m_accs_inst_wise,
        d_mrrs,
        d_top1_top1_accs,
        d_top3_top1_accs,
        d_top5_top1_accs,
        d_top5_top5_accs,
    )

    loss = losses.mean().item()
    m_loss = m_losses.mean().item()
    d_loss = d_losses.mean().item()
    inbatch_m_acc_cont_wise = inbatch_m_accs_cont_wise.mean().item()
    inbatch_m_acc_inst_wise = inbatch_m_accs_inst_wise.mean().item()
    d_mrr = d_mrrs.mean().item()
    d_top1_top1_acc = d_top1_top1_accs.float().mean().item()
    d_top3_top1_acc = d_top3_top1_accs.float().mean().item()
    d_top5_top1_acc = d_top5_top1_accs.float().mean().item()
    d_top5_top5_acc = d_top5_top5_accs.float().mean().item()

    return (
        loss,
        m_loss,
        d_loss,
        inbatch_m_acc_cont_wise,
        inbatch_m_acc_inst_wise,
        d_mrr,
        d_top1_top1_acc,
        d_top3_top1_acc,
        d_top5_top1_acc,
        d_top5_top5_acc,
    )


def main(rank, config_path: str, world_size: int):
    setup(rank, world_size)
    args = Flags(config_path).get()

    if args.is_nsml:
        from nsml import GPU_NUM, DATASET_NAME

        print(f"[+] GPU NUM: {GPU_NUM}, DATASET NAME: {DATASET_NAME}")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print_gpu_status(rank)

    if rank == 0:
        print(
            "[+] Ablation Study #2: Our Contrasitve Loss vs. Listwise Ranking Loss\n",
            "Hypothesis: Our Contrasitve Loss is better than the case of existed Listwise Ranking Loss based on LTR Theory\n",
            "            ('maybe' it is because our model can leverage cosine similarity based multimodal space\n",
            "             learned from the Semantic Matching task if we use cosine similarity based contrasitve loss.\n",
            "             But I do not have any clear idea to prove it)\n"
        )

    # load meta data
    train_matching, train_discrim, valid_matching, valid_discrim = load_meta_data(args)

    # txt_encoder
    train_txt_preprocessor = TextPreprocessor(
        args.backbone_txt, max_length=args.txt_max_length, dropout=args.word_dropout
    )
    valid_txt_preprocessor = TextPreprocessor(
        args.backbone_txt, max_length=args.txt_max_length, dropout=0.0
    )

    # img_transforms
    train_transforms = get_train_transforms(h=args.img_h, w=args.img_w)
    valid_transforms = get_eval_transforms(h=args.img_h, w=args.img_w)

    train_m_loader, train_d_loader = compose_dataloaders_ablation2(
        meta_matching=train_matching,
        meta_discrim=train_discrim,
        target=args.target,
        img_dir=args.img_dir,
        img_transforms=train_transforms,
        txt_preprocessor=train_txt_preprocessor,
        plan_attrs=args.plan_attrs,
        prod_attrs=args.prod_attrs,
        matching_size=args.matching_size,
        discrim_size=args.discrim_size,
        discrim_iter=args.discrim_iter,
        sampling_method=args.sampling_method,
        p_txt_aug=args.p_txt_aug,
        num_workers=args.num_workers,
        rank=rank,
    )

    valid_m_loader, valid_d_loader = compose_dataloaders_ablation2(
        meta_matching=valid_matching,
        meta_discrim=valid_discrim,
        target=args.target,
        img_dir=args.img_dir,
        img_transforms=valid_transforms,
        txt_preprocessor=valid_txt_preprocessor,
        plan_attrs=args.plan_attrs,
        prod_attrs=None,
        matching_size=args.matching_size,
        discrim_size=50,
        discrim_iter=world_size,
        sampling_method=args.sampling_method,
        p_txt_aug=0.0,
        num_workers=args.num_workers,
        rank=rank,
    )

    if rank == 0:
        print(
            "[+] Train Data Description\n",
            f"Matching Data Size: {len(train_matching):,d}\n",
            f"Discrim Data Size: {len(train_discrim):,d}\n",
            f"# Steps: {len(train_m_loader):,d}\n",
            f"# Samples for Semantic Matching per Step: {args.matching_size:,d}\n",
            f"# Samples for Preference Discrimination per Step: {args.discrim_size}\n",
            f"Sampling Iteration: {args.discrim_iter}\n",
            f"Sampling Method: {args.sampling_method}\n",
            f"Plan Attributes: {args.plan_attrs}\n",
            f"Prod Attributes: {args.prod_attrs}\n",
            f"Text Aug. Prob: {args.p_txt_aug}\n",
            f"Pretrained Tokenizer: {args.backbone_txt}\n",
            f"Word Dropout: {args.word_dropout}\n",
        )
        print(
            "[+] Valid Data Description\n",
            f"Matching Data Size: {len(valid_matching):,d}\n",
            f"Discrim Data Size: {len(valid_discrim):,d}\n",
            f"# Steps: {len(valid_d_loader):,d}\n",
            f"# Samples for Semantic Matching per Step: {args.matching_size:,d}\n",
            f"# Samples for Preference Discrimination per Step: {50}\n",
            f"Sampling Iteration: {1}\n",
            f"Plan Attributes: {args.plan_attrs}\n",
            f"Prod Attributes: {args.prod_attrs}\n",
            f"Pretrained Tokenizer: {args.backbone_txt}\n",
            f"Word Dropout: {0.0}\n",
        )

    # build model & train settings
    model = Ablation2(
        feature_dim=args.feature_dim,
        queue_size=args.matching_size,
        backbone_txt=args.backbone_txt,
        backbone_img=args.backbone_img,
        temperature=args.temperature,
        rank=rank,
    )
    model.to(device)
    model = DDP(model, device_ids=[rank])

    params_txt = [p for p in model.module.enc_context.parameters()]
    params_img_others = [p for p in model.module.enc_instance.parameters()] + [
        p for p in model.module.agg.parameters()
    ]

    num_tot_params = sum(j.numel() for j in [p for p in model.module.parameters()])
    num_txt_params = sum(
        j.numel() for j in [p for p in model.module.enc_context.parameters()]
    )
    num_img_params = sum(
        j.numel() for j in [p for p in model.module.enc_instance.parameters()]
    )
    num_other_params = num_tot_params - (num_txt_params + num_img_params)
    num_params_to_update = sum(
        j.numel() for j in [p for p in model.module.parameters() if p.requires_grad]
    )

    optim_txt = optim.AdamW(params_txt, lr=args.lr_txt)
    optim_img = optim.AdamW(params_img_others, lr=args.lr_img)

    # scheduler
    num_training_steps = len(train_d_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * 0.01)
    scheduler_txt = get_cosine_with_hard_restarts_schedule_with_warmup(
        optim_txt,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )
    scheduler_img = get_cosine_with_hard_restarts_schedule_with_warmup(
        optim_img,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )
    print_gpu_status(rank)

    if rank == 0:
        print(
            "[+] Model Description\n",
            f"Embedding Dimension: {args.feature_dim}\n",
            f"Queue Size: {args.queue_size}\n",
            f"Backbone for Text Encoder: {args.backbone_txt}\n",
            f"Backbone for Image Encoder: {args.backbone_img}\n",
            f"# Total Params: {num_tot_params:,d}\n",
            f"# Params for Text Encoder: {num_txt_params:,d}\n",
            f"# Params for Image Encoder: {num_img_params:,d}\n",
            f"# Params for the Others: {num_other_params:,d}\n",
        )
        print(
            "[+] Train Description\n",
            f"Epochs: {args.epochs}\n",
            f"LR for Text Encoder: {args.lr_txt}\n",
            f"LR for Image Encoder: {args.lr_img}\n",
            f"# Params to Optimize: {num_params_to_update:,d}\n",
            f"Temperature: {args.temperature}\n",
            f"Seed: {args.seed}\n",
        )

    dist.barrier()

    # load checkpoint
    start_epoch, start_step = 0, 0
    if args.ckpt_load_path is not None:
        (
            start_epoch,
            start_step,
            model,
            optim_txt,
            optim_img,
            scheduler_txt,
            scheduler_img,
        ) = load_checkpoint(
            model,
            optim_img,
            optim_txt,
            scheduler_img,
            scheduler_txt,
            args.ckpt_load_path,
            device=device,
        )
        print(
            f"[+] Checkpoint (rank: {rank}): {args.ckpt_load_path} loaded successfully :D\n",
            f"Start epoch from {start_epoch}/{args.epochs}\n",
            f"Start step from {start_step}/{args.epochs}\n",
        )

    # register arguments to track for logging
    logger = None  # None for rank != 0
    if rank == 0:
        # P.D. - Preference Discrimination
        # S.M. - Semantic Matching
        logger = Logger(args.log_save_dir)
        logger.register(
            [
                "epoch",
                "step",
                "lr_txt",
                "lr_img",
                "train_loss",  # overall loss
                "train_m_loss",  # loss of S.M.
                "train_d_loss",  # loss of P.D.
                "train_m_acc_cont_wise",  # S.M accuracy (context-wise)
                "train_m_acc_inst_wise",  # S.M accuracy (instance-wise)
                "train_mrr",  # P.D. MRR
                "train_top1top1_acc",  # P.D. Top1-Top1 accuracy
                "train_top3top1_acc",  # P.D. Top3-Top1 accuracy
                "train_top5top1_acc",  # P.D. Top5-Top1 accuracy
                "train_top5top5_acc",  # P.D. Top5-Top5 accuracy
                "valid_loss",  # overall loss
                "valid_m_loss",  # loss of S.M.
                "valid_d_loss",  # loss of P.D.
                "valid_m_acc_cont_wise",  # S.M accuracy (context-wise)
                "valid_m_acc_inst_wise",  # S.M accuracy (instance-wise)
                "valid_mrr",  # P.D. MRR
                "valid_top1top1_acc",  # P.D. Top1-Top1 accuracy
                "valid_top3top1_acc",  # P.D. Top3-Top1 accuracy
                "valid_top5top1_acc",  # P.D. Top5-Top1 accuracy
                "valid_top5top5_acc",  # P.D. Top5-Top5 accuracy
            ]
        )

    best_loss = 1e9
    best_d_loss = 1e9
    best_mrr = 0
    best_top1_top1_acc = 0
    best_top5_top1_acc = 0

    best_loss_path = os.path.join(args.ckpt_save_dir, "best_loss_clik.pth")
    best_d_loss_path = os.path.join(args.ckpt_save_dir, "best_dloss_clik.pth")
    best_mrr_path = os.path.join(args.ckpt_save_dir, "best_mrr_clik.pth")
    best_top1_top1_path = os.path.join(args.ckpt_save_dir, "best_top1top1_clik.pth")
    best_top5_top1_path = os.path.join(args.ckpt_save_dir, "best_top5top1_clik.pth")
    current_epoch_path = os.path.join(args.ckpt_save_dir, "current_epoch_clik.pth")

    scaler = GradScaler()
    dist.barrier()

    for epoch in range(start_epoch, args.epochs):

        # train
        train_one_epoch(
            model,
            train_m_loader,
            train_d_loader,
            optim_txt,
            optim_img,
            scheduler_txt,
            scheduler_img,
            cur_epoch=epoch,
            tot_epoch=args.epochs,
            scaler=scaler,
            device=device,
            logger=logger,
        )

        dist.barrier()

        # valid
        (
            valid_loss,
            valid_m_loss,
            valid_d_loss,
            valid_inbatch_m_acc_cont_wise,
            valid_inbatch_m_acc_inst_wise,
            valid_d_mrr,
            valid_d_top1_top1_acc,
            valid_d_top3_top1_acc,
            valid_d_top5_top1_acc,
            valid_d_top5_top5_acc,
        ) = valid_one_epoch(
            model,
            valid_m_loader,
            valid_d_loader,
            cur_epoch=epoch,
            tot_epoch=args.epochs,
            device=device,
        )

        dist.barrier()

        if rank == 0:
            # logging
            valid_log = dict(
                epoch=epoch,
                valid_loss=valid_loss,
                valid_m_loss=valid_m_loss,
                valid_d_loss=valid_d_loss,
                valid_m_acc_cont_wise=valid_inbatch_m_acc_cont_wise,
                valid_m_acc_inst_wise=valid_inbatch_m_acc_inst_wise,
                valid_mrr=valid_d_mrr,
                valid_top1top1_acc=valid_d_top1_top1_acc,
                valid_top3top1_acc=valid_d_top3_top1_acc,
                valid_top5top1_acc=valid_d_top5_top1_acc,
                valid_top5top5_acc=valid_d_top5_top5_acc,
            )
            logger.record(valid_log)

            # update & measure epoch result
            logger.update()
            ep_result = logger.return_last_logs()

            print(
                f"[+] Epoch: {epoch}/{args.epochs-1}\n",
                f"Train Loss: {ep_result['train_loss']:.4f} - Matching Loss: {ep_result['train_m_loss']:.4f}, Discrim Loss: {ep_result['train_d_loss']:.4f}\n",
                f"Train in-batch Matching ACC(Context-wise): {ep_result['train_m_acc_cont_wise']:.4f}\n",
                f"Train in-batch Matching ACC(Instance-wise): {ep_result['train_m_acc_inst_wise']:.4f}\n",
                f"Train Discrim MRR: {ep_result['train_mrr']:.4f}\n",
                f"Train Discrim Top1/Top1 ACC: {ep_result['train_top1top1_acc']:.4f}\n",
                f"Train Discrim Top3/Top1 ACC: {ep_result['train_top3top1_acc']:.4f}\n",
                f"Train Discrim Top5/Top1 ACC: {ep_result['train_top5top1_acc']:.4f}\n",
                f"Train Discrim Top5/Top5 ACC: {ep_result['train_top5top5_acc']:.4f}\n",
                f"Valid Loss: {ep_result['valid_loss']:.4f} - Matching Loss: {ep_result['valid_m_loss']:.4f}, Discrim Loss: {ep_result['valid_d_loss']:.4f}\n",
                f"Valid in-batch Matching ACC(Context-wise): {ep_result['valid_m_acc_cont_wise']:.4f}\n",
                f"Valid in-batch Matching ACC(Instance-wise): {ep_result['valid_m_acc_inst_wise']:.4f}\n",
                f"Valid Discrim MRR: {ep_result['valid_mrr']:.4f}\n",
                f"Valid Discrim Top1/Top1 ACC: {ep_result['valid_top1top1_acc']:.4f}\n",
                f"Valid Discrim Top3/Top1 ACC: {ep_result['valid_top3top1_acc']:.4f}\n",
                f"Valid Discrim Top5/Top1 ACC: {ep_result['valid_top5top1_acc']:.4f}\n",
                f"Valid Discrim Top5/Top5 ACC: {ep_result['valid_top5top5_acc']:.4f}\n",
            )

            # save checkpoint
            save_args = dict(
                args=args,
                epoch=epoch,
                step=(epoch + 1) * len(train_m_loader),
                model=model,
                optim_txt=optim_txt,
                optim_img=optim_img,
                scheduler_txt=scheduler_txt,
                scheduler_img=scheduler_img,
            )

            save_checkpoint(
                **save_args, save_path=current_epoch_path, is_distributed=True
            )

            if ep_result["valid_loss"] < best_loss:
                best_loss = ep_result["valid_loss"]
                save_checkpoint(**save_args, save_path=best_loss_path)
                print(f"[+] Best Loss: {best_loss:.4f} - model saved!")

            if ep_result["valid_d_loss"] < best_d_loss:
                best_d_loss = ep_result["valid_d_loss"]
                save_checkpoint(**save_args, save_path=best_d_loss_path)
                print(f"[+] Best Discrim Loss: {best_d_loss:.4f} - model saved!")

            if ep_result["valid_mrr"] > best_mrr:
                best_mrr = ep_result["valid_mrr"]
                save_checkpoint(**save_args, save_path=best_mrr_path)
                print(f"[+] Best MRR: {best_mrr:.4f} - model saved!")

            if ep_result["valid_top1top1_acc"] > best_top1_top1_acc:
                best_top1_top1_acc = ep_result["valid_top1top1_acc"]
                save_checkpoint(**save_args, save_path=best_top1_top1_path)
                print(
                    f"[+] Best Top1-Top1 ACC: {best_top1_top1_acc:.4f} - model saved!"
                )

            if ep_result["valid_top5top1_acc"] > best_top5_top1_acc:
                best_top5_top1_acc = ep_result["valid_top5top1_acc"]
                save_checkpoint(**save_args, save_path=best_top5_top1_path)
                print(
                    f"[+] Best Top5-Top1 ACC: {best_top5_top1_acc:.4f} - model saved!"
                )

            # save logs - it will be overwrited during train
            logger.save(save_dir=args.log_save_dir)

        dist.barrier()
