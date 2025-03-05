import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import zarr


input_dir = '/home/yusu/new_home/datasets/books3_splitted_doubletrain_finetune_tokenized/tokenizer_name-meta-llama/Llama-2-7b-hf-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False'
all_npy_files = [
    # 'train_part1.npy',
    # 'train_part2.npy',
    # 'finetune.npy',
    # 'test.npy',
    'validation.npy',
]

output_dir = '/home/yusu/new_home/datasets/books3_splitted_doubletrain_finetune_tokenized_Zarr/tokenizer_name-meta-llama/Llama-2-7b-hf-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False'
os.makedirs(output_dir, exist_ok=True)
output_file = osp.join(output_dir, 'data.zarr')

codec = zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle)

store = zarr.storage.LocalStore(output_file)

vocab_size = 32000

for k in tqdm(all_npy_files):

    name = k.split('.')[0]

    print(f"Creating array {name}")

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

store.close()


##################################################################################

# batch_size = (4 * 1024 * 1024 * 1024) * 2  # this number of int16 -> 4GB * 2 = 8GB
#
# with h5py.File(output_file, 'a') as f_out:
#
#     for npy_file in tqdm(all_npy_files, desc="Appending npy file"):
#
#         key = npy_file.split('.')[0]
#
#         print(f'Converting split: {key}')
#
#         if key not in f_out:
#             chunk_size = 2048
#             dset = f_out.create_dataset(
#                 key,
#                 shape=(0,),
#                 maxshape=(None,),
#                 dtype='uint16',
#                 chunks=(chunk_size,),
#                 compression='gzip',
#             )
#         else:
#             dset = f_out[key]
#
#         data = np.load(osp.join(input_dir, npy_file), mmap_mode='r')
#         total_size = data.shape[0]
#
#         # Process the data in chunks
#         for start in tqdm(range(0, total_size, batch_size), "Processing chunks"):
#             end = min(start + batch_size, total_size)
#             chunk = data[start:end]  # Only this chunk is loaded into memory
#
#             # Resize and append chunk
#             dset.resize(dset.shape[0] + chunk.shape[0], axis=0)
#             dset[-chunk.shape[0]:] = chunk
#
#             # Ensure the chunk is released from memory immediately
#             del chunk
#             gc.collect()
#
#         del data
#         gc.collect()
#
#         print(f'Total tokens in {key}: {dset.shape[0]}')
#
#     if "tokenizer" not in f_out:
#         with open(osp.join(input_dir, "tokenizer.pkl"), "rb") as f:
#             tokenizer = pickle.load(f)
#         tokenizer_bytes = pickle.dumps(tokenizer)
#         tokenizer_np = np.void(tokenizer_bytes)
#         f_out.create_dataset("tokenizer", data=tokenizer_np)
#
#         ## Example to extract tokenizer
#         # with h5py.File("tokenizer.h5", "r") as hf:
#         #     tokenizer_np = hf["tokenizer"][()]
#         #     tokenizer = pickle.loads(tokenizer_np.tobytes())  # Convert back to Python object
