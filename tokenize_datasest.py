"""
@Xinhao
c3d-standard-180 180 vCPUs 720GB mem --> num_workers=16
"""
import os
from pathlib import Path
import argparse
import shutil
import subprocess
import torch
import numpy as np

from language_modeling_hf import LMDataModule


def div_up(x: int, y: int) -> int:
    return (x + y - 1) // y

def num_cpu_cores():
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except ImportError:
        return len(os.sched_getaffinity(0))


def tokenize_books():
    batch_size = 8
    dataset_name = 'books3_splitted_finetune'  # useless
    dataset_config_name = None
    cache_dir = Path('/mnt/disks/persistent/books3_splitted_finetune')  # path to save tokenized dataset
    raw_json_path = '/mnt/disks/persistent/lwm_raw/lwm_text_data/combined_books.jsonl'
    finetune_ratio = 0.167  # 1/6 of full train set becomes finetune set, 5/6 is pre-train set
    max_length = 2048  # only useful for deciding chunking data for sampler idx, won't affect tokenization
    num_workers = num_cpu_cores() // 2
    datamodule = LMDataModule(
        dataset_name,
        tokenizer_name='meta-llama/Llama-2-7b-hf',
        dataset_config_name=dataset_config_name,
        max_length=max_length,
        cache_dir=cache_dir,
        add_eos=True,
        batch_size=batch_size,
        num_workers=num_workers,
        raw_json_path=raw_json_path,
        finetune_ratio=finetune_ratio
    )
    datamodule.prepare_data()
    datamodule.setup(stage='fit')

def tokenize_dclm():
    dataset_name = '/persistent_dclm/datasets/dclm_200B_raw'
    dataset_config_name = None
    cache_dir = Path('/persistent_dclm/datasets/dclm_200B_tok_la2')
    num_workers = num_cpu_cores() // 2
    val_ratio = 0.05 # 5B out of 200B: 0.025 -> 0.05 to be safe
    datamodule = LMDataModule(
        dataset_name,
        tokenizer_name='meta-llama/Llama-2-7b-hf',
        dataset_config_name=dataset_config_name,
        cache_dir=cache_dir,
        add_eos=True,
        num_workers=num_workers,
        raw_json_path=None,
        val_ratio=val_ratio
    )
    datamodule.prepare_data()
    datamodule.setup(stage='fit')


tokenize_fn_dict = {
    "books": tokenize_books,
    "dclm": tokenize_dclm
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="dclm")
    args = parser.parse_args()

    assert args.dataset in tokenize_fn_dict.keys()

    tokenize_fn_dict[args.dataset]()
