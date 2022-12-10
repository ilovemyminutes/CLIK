import json
import os
import pickle
import random
from datetime import datetime, timedelta

import numpy as np
import torch
from psutil import virtual_memory


def set_seed(seed: int = 27):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def print_system_envs():
    """시스템 환경을 출력"""
    num_gpus = torch.cuda.device_count()
    num_cpus = os.cpu_count()
    cpu_mem_size = virtual_memory().available // (1024**3)
    gpu_mem_size = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(
        "[+] System environments\n",
        f"Number of GPUs : {num_gpus}\n",
        f"Number of CPUs : {num_cpus}\n",
        f"CPU Memory Size : {cpu_mem_size:.4f} GB\n",
        f"GPU Memory Size : {gpu_mem_size:.4f} GB\n",
    )


def print_gpu_status(gpu_idx: int = 3) -> None:
    """GPU 이용 상태를 출력"""
    total_mem = torch.cuda.get_device_properties(gpu_idx).total_memory / 1024**3
    reserved = torch.cuda.memory_reserved(gpu_idx) / 1024**3
    allocated = torch.cuda.memory_allocated(gpu_idx) / 1024**3
    free = reserved - allocated
    print(
        "[+] GPU Status\n",
        f"Index: {gpu_idx}\n",
        f"Total: {total_mem:.4f} GB\n",
        f"Reserved: {reserved:.4f} GB\n",
        f"Allocated: {allocated:.4f} GB\n",
        f"Residue: {free:.4f} GB\n",
    )


def save_json(path: str, f: object) -> None:
    with open(path, "w", encoding="utf-8") as json_path:
        json.dump(f, json_path, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as json_file:
        output = json.load(json_file)
    return output


def load_pickle(path: str):
    with open(path, "rb") as pkl_file:
        output = pickle.load(pkl_file)
    return output


def save_pickle(path: str, f: object) -> None:
    with open(path, "wb") as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_timestamp(date_format: str = "%Y%m%d%H%M"):
    stamp = datetime.now() + timedelta(hours=9)
    return stamp.strftime(date_format)
