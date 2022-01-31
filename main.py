# nsml: silkstaff/thumbnail-selection:2.0
import os
import argparse
from importlib import import_module
import torch.multiprocessing as mp
from utils import get_timestamp, Flags

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # to debug DDP in detail


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="./configs/CLIK.yaml")
    args = parser.parse_args()

    config_path = args.config_path
    assert os.path.isfile(config_path), f"'{config_path}' not found"

    # load config file
    args = Flags(config_path).get()

    if args.is_nsml:
        os.environ["HOME"] = "/home/nsml"

    # make directory to save logs & checkpoints
    os.makedirs(args.log_save_dir, exist_ok=True)
    os.makedirs(args.ckpt_save_dir, exist_ok=True)

    # save simple time stamp info flag file
    timestamp = f"exp{get_timestamp()}_{args.exp_title}"
    f = open(os.path.join(args.log_save_dir, timestamp), "w")
    f.close()
    f = open(os.path.join(args.ckpt_save_dir, timestamp), "w")
    f.close()

    name = args.network.lower()
    train_module = (
        f"train.train_{name}" if not args.is_distributed else f"train.train_{name}_dist"
    )
    train_fn = getattr(import_module(train_module), "main")

    # verbose arguments
    print("=" * 100)
    print(args)
    print("=" * 100)

    # start train
    if not args.is_distributed:
        train_fn(args)
    else:
        assert (
            args.matching_size % args.world_size == 0
        ), f"matching_size({args.matching_size}) needs to be divided by world_size({args.world_size})"
        assert (
            args.discrim_iter % args.world_size == 0
        ), f"discrim_iter({args.discrim_iter}) needs to be divided by world_size({args.world_size})"
        print(
            f"[+] Distributed Training\n",
            f"world_size: {args.world_size}\n",
            f"num_workers: {args.num_workers}\n",
        )
        mp.spawn(
            train_fn,
            args=(config_path, args.world_size),
            nprocs=args.world_size,
            join=True,
        )
