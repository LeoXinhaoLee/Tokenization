import os
import os.path as osp
import argparse
from tqdm import tqdm
import numpy as np
import zarr

print(zarr.__version__)

parser = argparse.ArgumentParser(description="Convert npy to zarr")
parser.add_argument("--input_dir", type=str, default='/home/yusu/new_home/datasets/books3_splitted_doubletrain_finetune_tokenized/tokenizer_name-meta-llama/Llama-2-7b-hf-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False')
parser.add_argument("--output_dir", type=str, default='/home/yusu/new_home/datasets/books3_splitted_doubletrain_finetune_tokenized_Zarr/tokenizer_name-meta-llama/Llama-2-7b-hf-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False')
args = parser.parse_args()

input_dir = args.input_dir
all_npy_files = [
    'train_part1.npy',
    'train_part2.npy',
    'finetune.npy',
    'test.npy',
    'validation.npy',
]

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
output_file = osp.join(output_dir, 'data.zarr')

store = zarr.storage.LocalStore(output_file)

codec = zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle)

vocab_size = 32000

for k in tqdm(all_npy_files):
    name = k.split('.')[0]
    print(f"Creating array {name}")
    
    # Load data
    src_data = np.load(osp.join(input_dir, k), mmap_mode='r')
    
    a = zarr.create_array(
        name=name,
        store=store,
        # shape=(),
        data=src_data,
        chunks=(2**19,),  # 512KB per chunk
        # dtype="u2",
        compressors=codec,
        dimension_names=["token"],
        overwrite=False,
    )
    
    a.attrs["vocab_size"] = vocab_size
