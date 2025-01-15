import pdb
import os
import os.path as osp
from pathlib import Path
import glob
import random
from tqdm import tqdm
import gc
import numpy as np
import h5py


all_npy_files = [
    '/home/yusu/new_home/datasets/dclm_200B_tok_la2/tokenizer_name-meta-llama/Llama-2-7b-hf-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False/train_part1.npy',
]

output_dir = '/home/yusu/new_home/datasets/dclm_200B_tok_la2_h5/tokenizer_name-meta-llama/Llama-2-7b-hf-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False'
os.makedirs(output_dir, exist_ok=True)
output_file = osp.join(output_dir, 'train_part1.h5')

batch_size = (4 * 1024 * 1024 * 1024) * 2  # this number of int16 -> 4GB * 2 = 8GB

with h5py.File(output_file, 'a') as f_out:
    if 'train_dataset' not in f_out:
        # dset = f_out.create_dataset('train_dataset', shape=(0,), maxshape=(None,), dtype='int16')  # la2 tokenizer vocab: int16
        chunk_size = 2048
        dset = f_out.create_dataset(
            'train_dataset',
            shape=(0,),
            maxshape=(None,),
            dtype='int16',
            chunks=(chunk_size,),
            compression='gzip',
        )
    else:
        dset = f_out['train_dataset']

    print(f'Existing token: {dset.shape[0]}')

    token_count = 0
    for npy_file in tqdm(all_npy_files, desc="Appending npy file"):
        data = np.load(npy_file, mmap_mode='r')
        total_size = data.shape[0]

        # Process the data in chunks
        for start in tqdm(range(0, total_size, batch_size), "Processing chunks"):
            end = min(start + batch_size, total_size)
            chunk = data[start:end]  # Only this chunk is loaded into memory

            # Resize and append chunk
            dset.resize(dset.shape[0] + chunk.shape[0], axis=0)
            dset[-chunk.shape[0]:] = chunk
            token_count += chunk.shape[0]

            # Ensure the chunk is released from memory immediately
            del chunk
            gc.collect()

        del data
        gc.collect()

print(f'Add token: {token_count}')
print(f'Data saved to {output_file}')
