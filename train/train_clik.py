import os
from tqdm import tqdm
import warnings
import numpy as np
import torch
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from networks import CLIK
from preprocessing import (
    get_train_transforms,
    get_eval_transforms,
    TextPreprocessor,
)
from utils.data_utils import compose_dataloaders, load_meta_data, compose_batch
from utils.flags import Flags
from utils.metric import accuracy, mean_reciprocal_rank, topn_isin_topk
from utils.logger import Logger
from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils import print_gpu_status, set_seed


def train_one_epoch(
    model: CLIK,
    m_loader: DataLoader,
    d_loader: DataLoader,
    optim_txt: Optimizer,
    optim_img: Optimizer,
    scheduler_txt: LambdaLR,
    scheduler_img: LambdaLR,
    cur_epoch: int,
    tot_epoch: int,
    logger: Logger,
    scaler: GradScaler,
    device: torch.device,
):
    model.train()
    for step, ((m_topics, m_images), (r_topics, r_images)) in tqdm(
        enumerate(zip(m_loader, d_loader)), desc=f"[Train: {cur_epoch}/{tot_epoch-1}]"
    ):
        batch_matching = compose_batch(m_topics, m_images, device=device)
        batch_ranking = compose_batch(r_topics, r_images, device=device)
        with autocast(enabled=True):
            (
                m_logits_topic_wise,
                m_logits_image_wise,
                m_labels,
                m_loss,
                r_logits,
                _,
                r_loss,
            ) = model(batch_matching, batch_ranking)
            loss = 20 * m_loss + r_loss

        optim_txt.zero_grad()
        optim_img.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optim_txt)
        scaler.step(optim_img)
        scaler.update()
        scheduler_txt.step()
        scheduler_img.step()

        m_acc_topic_wise = accuracy(m_logits_topic_wise, m_labels)
        m_acc_image_wise = accuracy(m_logits_image_wise, m_labels)
        r_mrr = mean_reciprocal_rank(r_logits)
        r_top1_top1_acc = topn_isin_topk(r_logits, n=1, k=1)
        r_top3_top1_acc = topn_isin_topk(r_logits, n=3, k=1)
        r_top5_top1_acc = topn_isin_topk(r_logits, n=5, k=1)
        r_top5_top5_acc = topn_isin_topk(r_logits, n=5, k=5)

        train_log = dict(
            epoch=cur_epoch,
            step=step + (cur_epoch * len(m_loader)),
            lr_txt=scheduler_txt.get_last_lr()[0],
            lr_img=scheduler_img.get_last_lr()[0],
            train_loss=loss.item(),
            train_m_loss=m_loss.item(),
            train_r_loss=r_loss.item(),
            train_m_acc_topic_wise=m_acc_topic_wise,
            train_m_acc_image_wise=m_acc_image_wise,
            train_mrr=r_mrr,
            train_top1top1_acc=r_top1_top1_acc,
            train_top3top1_acc=r_top3_top1_acc,
            train_top5top1_acc=r_top5_top1_acc,
            train_top5top5_acc=r_top5_top5_acc,
        )

        logger.record(train_log)


def valid_one_epoch(
    model: CLIK,
    m_loader: DataLoader,
    r_loader: DataLoader,
    cur_epoch: int,
    tot_epoch: int,
    device: torch.device,
):
    model.eval()

    losses = []  # overall loss
    m_losses = []  # loss of S.M.
    r_losses = []  # loss of P.D.
    inbatch_m_accs_topic_wise = []  # S.M accuracy (Topic-wise)
    inbatch_m_accs_image_wise = []  # S.M accuracy (Image-wise)
    r_mrrs = []  # P.D. MRR
    r_top1_top1_accs = []  # P.D. Top1-Top1 accuracy
    r_top3_top1_accs = []  # P.D. Top3-Top1 accuracy
    r_top5_top1_accs = []  # P.D. Top5-Top1 accuracy
    r_top5_top5_accs = []  # P.D. Top5-Top5 accuracy

    for (m_topics, m_images), (r_topics, r_images) in tqdm(
        zip(m_loader, r_loader), desc=f"[Valid: {cur_epoch}/{tot_epoch-1}]"
    ):
        batch_matching = compose_batch(m_topics, m_images, device=device)
        batch_ranking = compose_batch(r_topics, r_images, device=device)
        with autocast(enabled=True):
            with torch.no_grad():
                (
                    m_logits_topic_wise,
                    m_logits_image_wise,
                    m_labels,
                    m_loss,
                    r_logits,
                    _,
                    r_loss,
                ) = model(batch_matching, batch_ranking, update_queue=False)

            loss = 20 * m_loss + r_loss

        m_acc_topic_wise = accuracy(m_logits_topic_wise, m_labels)
        m_acc_image_wise = accuracy(m_logits_image_wise, m_labels)
        ranking_mrr = mean_reciprocal_rank(r_logits)
        ranking_top1_top1_acc = topn_isin_topk(r_logits, n=1, k=1)
        ranking_top3_top1_acc = topn_isin_topk(r_logits, n=3, k=1)
        ranking_top5_top1_acc = topn_isin_topk(r_logits, n=5, k=1)
        ranking_top5_top5_acc = topn_isin_topk(r_logits, n=5, k=5)

        losses.append(loss.item())
        m_losses.append(m_loss.item())
        r_losses.append(r_loss.item())
        inbatch_m_accs_topic_wise.append(m_acc_topic_wise)
        inbatch_m_accs_image_wise.append(m_acc_image_wise)
        r_mrrs.append(ranking_mrr)
        r_top1_top1_accs.extend(ranking_top1_top1_acc)
        r_top3_top1_accs.extend(ranking_top3_top1_acc)
        r_top5_top1_accs.extend(ranking_top5_top1_acc)
        r_top5_top5_accs.extend(ranking_top5_top5_acc)

    loss = np.mean(losses)
    m_loss = np.mean(m_losses)
    r_loss = np.mean(r_losses)
    inbatch_m_acc_topic_wise = np.mean(inbatch_m_accs_topic_wise)
    inbatch_m_acc_image_wise = np.mean(inbatch_m_accs_image_wise)
    r_mrr = np.mean(r_mrrs)
    r_top1_top1_acc = np.mean(r_top1_top1_accs)
    r_top3_top1_acc = np.mean(r_top3_top1_accs)
    r_top5_top1_acc = np.mean(r_top5_top1_accs)
    r_top5_top5_acc = np.mean(r_top5_top5_accs)

    return (
        loss,
        m_loss,
        r_loss,
        inbatch_m_acc_topic_wise,
        inbatch_m_acc_image_wise,
        r_mrr,
        r_top1_top1_acc,
        r_top3_top1_acc,
        r_top5_top1_acc,
        r_top5_top5_acc,
    )


def main(args: Flags):
    device = torch.device(
        f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu"
    )
    set_seed(args.seed)
    print_gpu_status(args.gpu_idx)

    # load meta data
    train_matching, train_ranking, valid_matching, valid_ranking = load_meta_data(args)

    train_cats = train_ranking[args.main_cat_depth].unique().tolist()
    valid_cats = valid_ranking[args.main_cat_depth].unique().tolist()
    if len(set(train_cats).intersection(set(valid_cats))) != len(train_cats):
        warnings.warn("There's a category in valid data which is not in train data")

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

    train_m_loader, train_r_loader = compose_dataloaders(
        meta_matching=train_matching,
        meta_Ranking=train_ranking,
        labeling_criterion=args.labeling_criterion,
        img_dir=args.img_dir,
        img_transforms=train_transforms,
        txt_preprocessor=train_txt_preprocessor,
        exhibit_attrs=args.exhibit_attrs,
        prod_attrs=args.prod_attrs,
        matching_size=args.matching_size,
        ranking_size=args.ranking_size,
        ranking_iter=args.ranking_iter,
        sampling_method=args.sampling_method,
        txt_aug_prob=args.txt_aug_prob,
        num_workers=args.num_workers,
    )

    valid_m_loader, valid_d_loader = compose_dataloaders(
        meta_matching=valid_matching,
        meta_Ranking=valid_ranking,
        labeling_criterion=args.labeling_criterion,
        img_dir=args.img_dir,
        img_transforms=valid_transforms,
        txt_preprocessor=valid_txt_preprocessor,
        exhibit_attrs=args.exhibit_attrs,
        prod_attrs=None,
        matching_size=args.matching_size,
        ranking_size=args.ranking_size,
        ranking_iter=args.ranking_iter,
        sampling_method=args.sampling_method,
        txt_aug_prob=args.txt_aug_prob,
        num_workers=args.num_workers,
    )
    print(
        "[+] Train Data Description\n",
        f"Matching Data Size: {len(train_matching):,d}\n",
        f"Ranking Data Size: {len(train_ranking):,d}\n",
        f"# Steps: {len(train_r_loader):,d}\n",
        f"# Samples for Semantic Matching per Step: {args.matching_size:,d}\n",
        f"# Samples for Preference Rankingination per Step: {args.Ranking_size}\n",
        f"Sampling Iteration: {args.Ranking_iter}\n",
        f"Sampling Method: {args.sampling_method}\n",
        f"Exhibition Attributes: {args.exhibit_attrs}\n",
        f"Prod Attributes: {args.prod_attrs}\n",
        f"Text Aug. Prob: {args.txt_aug_prob}\n",
        f"Pretrained Tokenizer: {args.backbone_txt}\n",
        f"Word Dropout: {args.word_dropout}\n",
    )
    print(
        "[+] Valid Data Description\n",
        f"Matching Data Size: {len(valid_matching):,d}\n",
        f"Ranking Data Size: {len(valid_ranking):,d}\n",
        f"# Steps: {len(valid_d_loader):,d}\n",
        f"# Samples for Semantic Matching per Step: {args.matching_size:,d}\n",
        f"# Samples for Preference Rankingination per Step: {50}\n",
        f"Sampling Iteration: {1}\n",
        f"Exhibition Attributes: {args.exhibit_attrs}\n",
        f"Prod Attributes: {args.prod_attrs}\n",
        f"Pretrained Tokenizer: {args.backbone_txt}\n",
        f"Word Dropout: {0.0}\n",
    )

    # build model & train settings
    model = CLIK(
        feature_dim=args.feature_dim,
        memory_bank_size=args.matching_size,
        backbone_txt=args.backbone_txt,
        backbone_img=args.backbone_img,
        temperature=args.temperature,
    )
    model.to(device)

    params_txt = [p for p in model.enc_context.parameters()]
    params_img_others = [p for p in model.enc_instance.parameters()] + [
        p for p in model.agg.parameters()
    ]

    num_tot_params = sum(j.numel() for j in [p for p in model.parameters()])
    num_txt_params = sum(j.numel() for j in [p for p in model.enc_context.parameters()])
    num_img_params = sum(
        j.numel() for j in [p for p in model.enc_instance.parameters()]
    )
    num_other_params = num_tot_params - (num_txt_params + num_img_params)
    num_params_to_update = sum(
        j.numel() for j in [p for p in model.parameters() if p.requires_grad]
    )

    optim_txt = optim.AdamW(params_txt, lr=args.lr_txt)
    optim_img = optim.AdamW(params_img_others, lr=args.lr_img)

    # scheduler
    num_training_steps = len(train_r_loader) * args.epochs
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
    print_gpu_status(args.gpu_idx)

    print(
        "[+] Model Description\n",
        f"Embedding Dimension: {args.feature_dim}\n",
        f"Memory Bank Size: {args.memory_bank_size}\n",
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
            f"[+] Checkpoint: {args.ckpt_load_path} loaded successfully :D\n",
            f"Start epoch from {start_epoch}/{args.epochs}\n",
            f"Start step from {start_step}/{args.epochs}\n",
        )

    # register arguments to track
    # P.D. - Preference Rankingination
    # S.M. - Semantic Matching
    logger = Logger(args.log_save_dir)
    logger.register(
        [
            "epoch",
            "step",
            "lr_txt",
            "lr_img",
            # train
            "train_loss",  # overall loss
            "train_m_loss",  # loss of S.M.
            "train_r_loss",  # loss of P.D.
            "train_m_acc_topic_wise",  # S.M accuracy (Topic-wise)
            "train_m_acc_image_wise",  # S.M accuracy (Image-wise)
            "train_mrr",  # P.D. MRR
            "train_top1top1_acc",  # P.D. Top1-Top1 accuracy
            "train_top3top1_acc",  # P.D. Top3-Top1 accuracy
            "train_top5top1_acc",  # P.D. Top5-Top1 accuracy
            "train_top5top5_acc",  # P.D. Top5-Top5 accuracy
            # valid
            "valid_loss",  # overall loss
            "valid_m_loss",  # loss of S.M.
            "valid_r_loss",  # loss of P.D.
            "valid_m_acc_topic_wise",  # S.M accuracy (Topic-wise)
            "valid_m_acc_image_wise",  # S.M accuracy (Image-wise)
            "valid_mrr",  # P.D. MRR
            "valid_top1top1_acc",  # P.D. Top1-Top1 accuracy
            "valid_top3top1_acc",  # P.D. Top3-Top1 accuracy
            "valid_top5top1_acc",  # P.D. Top5-Top1 accuracy
            "valid_top5top5_acc",  # P.D. Top5-Top5 accuracy
        ]
    )

    best_loss = 1e9
    best_r_loss = 1e9
    best_mrr = 0
    best_top1_top1_acc = 0
    best_top5_top1_acc = 0

    best_loss_path = os.path.join(args.ckpt_save_dir, "best_loss_clik.pth")
    best_r_loss_path = os.path.join(args.ckpt_save_dir, "best_rloss_clik.pth")
    best_mrr_path = os.path.join(args.ckpt_save_dir, "best_mrr_clik.pth")
    best_top1_top1_path = os.path.join(args.ckpt_save_dir, "best_top1top1_clik.pth")
    best_top5_top1_path = os.path.join(args.ckpt_save_dir, "best_top5top1_clik.pth")
    current_epoch_path = os.path.join(args.ckpt_save_dir, "current_epoch_clik.pth")

    scaler = GradScaler()

    for epoch in range(start_epoch, args.epochs):

        # train
        train_one_epoch(
            model,
            train_m_loader,
            train_r_loader,
            optim_txt,
            optim_img,
            scheduler_txt,
            scheduler_img,
            cur_epoch=epoch,
            tot_epoch=args.epochs,
            logger=logger,
            scaler=scaler,
            device=device,
        )

        # valid
        (
            valid_loss,
            valid_m_loss,
            valid_r_loss,
            valid_inbatch_m_acc_topic_wise,
            valid_inbatch_m_acc_image_wise,
            valid_r_mrr,
            valid_r_top1_top1_acc,
            valid_r_top3_top1_acc,
            valid_r_top5_top1_acc,
            valid_r_top5_top5_acc,
        ) = valid_one_epoch(
            model,
            valid_m_loader,
            valid_d_loader,
            cur_epoch=epoch,
            tot_epoch=args.epochs,
            device=device,
        )

        # logging
        valid_log = dict(
            epoch=epoch,
            valid_loss=valid_loss,
            valid_m_loss=valid_m_loss,
            valid_r_loss=valid_r_loss,
            valid_m_acc_topic_wise=valid_inbatch_m_acc_topic_wise,
            valid_m_acc_image_wise=valid_inbatch_m_acc_image_wise,
            valid_mrr=valid_r_mrr,
            valid_top1top1_acc=valid_r_top1_top1_acc,
            valid_top3top1_acc=valid_r_top3_top1_acc,
            valid_top5top1_acc=valid_r_top5_top1_acc,
            valid_top5top5_acc=valid_r_top5_top5_acc,
        )
        logger.record(valid_log)

        # get logs for each epoch
        logger.update()
        ep_result = logger.return_last_logs()  # result for each epoch

        print(
            f"[+] Epoch: {epoch}/{args.epochs-1}\n",
            f"Train Loss: {ep_result['train_loss']:.4f} - Matching Loss: {ep_result['train_m_loss']:.4f}, Ranking Loss: {ep_result['train_r_loss']:.4f}\n",
            f"Train in-batch Matching ACC(Topic-wise): {ep_result['train_m_acc_topic_wise']:.4f}\n",
            f"Train in-batch Matching ACC(Image-wise): {ep_result['train_m_acc_image_wise']:.4f}\n",
            f"Train Ranking MRR: {ep_result['train_mrr']:.4f}\n",
            f"Train Ranking Top1/Top1 ACC: {ep_result['train_top1top1_acc']:.4f}\n",
            f"Train Ranking Top3/Top1 ACC: {ep_result['train_top3top1_acc']:.4f}\n",
            f"Train Ranking Top5/Top1 ACC: {ep_result['train_top5top1_acc']:.4f}\n",
            f"Train Ranking Top5/Top5 ACC: {ep_result['train_top5top5_acc']:.4f}\n",
            f"Valid Loss: {ep_result['valid_loss']:.4f} - Matching Loss: {ep_result['valid_m_loss']:.4f}, Ranking Loss: {ep_result['valid_r_loss']:.4f}\n",
            f"Valid in-batch Matching ACC(Topic-wise): {ep_result['valid_m_acc_topic_wise']:.4f}\n",
            f"Valid in-batch Matching ACC(Image-wise): {ep_result['valid_m_acc_image_wise']:.4f}\n",
            f"Valid Ranking MRR: {ep_result['valid_mrr']:.4f}\n",
            f"Valid Ranking Top1/Top1 ACC: {ep_result['valid_top1top1_acc']:.4f}\n",
            f"Valid Ranking Top3/Top1 ACC: {ep_result['valid_top3top1_acc']:.4f}\n",
            f"Valid Ranking Top5/Top1 ACC: {ep_result['valid_top5top1_acc']:.4f}\n",
            f"Valid Ranking Top5/Top5 ACC: {ep_result['valid_top5top5_acc']:.4f}\n",
        )

        # save checkpoint
        save_args = dict(
            epoch=start_epoch + epoch,
            step=len(train_m_loader) * (start_epoch + epoch + 1),
            model=model,
            optim_txt=optim_txt,
            optim_img=optim_img,
            scheduler_txt=scheduler_txt,
            scheduler_img=scheduler_img,
        )

        save_checkpoint(**save_args, save_path=current_epoch_path, is_distributed=False)

        if ep_result["valid_loss"] < best_loss:
            best_loss = ep_result["valid_loss"]
            save_checkpoint(**save_args, save_path=best_loss_path)
            print(f"[+] Best Loss: {best_loss:.4f} - model saved!")

        if ep_result["valid_r_loss"] < best_r_loss:
            best_r_loss = ep_result["valid_r_loss"]
            save_checkpoint(**save_args, save_path=best_r_loss_path)
            print(f"[+] Best Ranking Loss: {best_r_loss:.4f} - model saved!")

        if ep_result["valid_mrr"] > best_mrr:
            best_mrr = ep_result["valid_mrr"]
            save_checkpoint(**save_args, save_path=best_mrr_path)
            print(f"[+] Best MRR: {best_mrr:.4f} - model saved!")

        if ep_result["valid_top1top1_acc"] > best_top1_top1_acc:
            best_top1_top1_acc = ep_result["valid_top1top1_acc"]
            save_checkpoint(**save_args, save_path=best_top1_top1_path)
            print(f"[+] Best Top1-Top1 ACC: {best_top1_top1_acc:.4f} - model saved!")

        if ep_result["valid_top5top1_acc"] > best_top5_top1_acc:
            best_top5_top1_acc = ep_result["valid_top5top1_acc"]
            save_checkpoint(**save_args, save_path=best_top5_top1_path)
            print(f"[+] Best Top5-Top1 ACC: {best_top5_top1_acc:.4f} - model saved!")
