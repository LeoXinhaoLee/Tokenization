import zarr
import os.path as osp
import numpy as np

# Path to your zarr file
output_dir = '/home/yusu/new_home/datasets/books3_splitted_doubletrain_finetune_tokenized_Zarr/tokenizer_name-meta-llama/Llama-2-7b-hf-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False'
zarr_path = osp.join(output_dir, 'data.zarr')

# Open the zarr store
zarr_group = zarr.open_group(zarr_path, mode='r')
# z = zarr.open_array(zarr_path, mode='r')
z = zarr_group['validation.npy']
print(z.shape, z[:10].dtype, z[:10], z[-10:])

print(z.attrs['vocab_size'])