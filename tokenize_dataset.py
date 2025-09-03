# /// script
# dependencies = [
#     "zstandard>=0.24.0",
#     "tqdm>=4.67.1",
#     "pytest==8.3.2",
#     "python-dotenv==1.0.1",
#     "pytorch-lightning==2.4.0",
#     "torch>=2.1.1",
#     "transformers==4.44.1",
#     "datasets==2.18.0",
#     "tokenizers==0.19.1",
#     "numpy==1.26.4",
#     "huggingface_hub==0.24.6",
#     "multiprocess==0.70.16",
#     "pyarrow== 17.0.0",
#     "pyarrow-hotfix==0.6",
#     "pandas==2.2.2",
#     "ray",
#     "h5py",
# ]
# ///
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


def tokenize_books(part):
    del part
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

def tokenize_dclm(part):
    dataset_name = f'/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yusu/datasets/dclm-600B-process/DCLM-600B-RAW-text/part_{part}'
    dataset_config_name = None
    cache_dir = Path(f'/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yusu/datasets/dclm-600B-process/DCLM-600B_tok_la2/part_{part}')
    num_workers = num_cpu_cores() // 2
    val_ratio = 0.03  # 20B out of 700B: 0.028 -> 0.03 to be safe
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
    parser.add_argument('--part', type=str, default="0")
    args = parser.parse_args()

    assert args.dataset in tokenize_fn_dict.keys()

    tokenize_fn_dict[args.dataset](args.part)
