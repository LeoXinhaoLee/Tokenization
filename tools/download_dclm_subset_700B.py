'''
After generating links.txt
(1) Extend output file name with it's global and local shard id. Otherwise will overwrite each other.
awk '
{
    url = $0;
    sub("\\?download=true", "", url);
    split(url, parts, "/");
    filename = parts[length(parts)-2] "_" parts[length(parts)-1] "_" parts[length(parts)];
    print url, "/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yusu/datasets/dclm-600B-process/DCLM-600B-RAW/" filename;
}' links.txt > processed_links.txt

(2) Download all items with 8 processes
xargs -n 2 -P 8 sh -c 'wget -O "$1" "$0"' < processed_links.txt
'''
import pdb
from pathlib import Path
import math
import random
import numpy as np

random.seed(0)
np.random.seed(0)

num_global_shards = 10
num_local_shards = 10
num_json = 279

num_total_file = num_global_shards * num_local_shards * num_json
print(f'Total shards number: {num_total_file}')

file_abs_id = [i for i in range(num_total_file)]
sampled_200B_abs_id = random.sample(file_abs_id, math.ceil(num_total_file / 20))  # 4T tokens -> 200B tokens
print(f'Sample 200B shards number: {len(sampled_200B_abs_id)}')

remaining_file_abs_id = list(set(file_abs_id) - set(sampled_200B_abs_id))
sampled_500B_abs_id = random.sample(remaining_file_abs_id, math.ceil(num_total_file / 8))  # 4T tokens -> 500B tokens
print(f'Sample 500B shards number: {len(sampled_500B_abs_id)}')

sampled_abs_id = sampled_200B_abs_id + sampled_500B_abs_id
print(f'Sample 700B shards number: {len(sampled_abs_id)}')
assert len(sampled_abs_id) == len(set(sampled_abs_id)), "File abs IDs have duplication!"

link_format = 'https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/resolve/main/global-shard_{:02}_of_10/local-shard_{}_of_10/shard_{:08}_processed.jsonl.zst'

link_list = []
for abs_id in sampled_abs_id:
    global_id = abs_id // (num_local_shards * num_json)
    local_id = (abs_id % (num_local_shards * num_json)) // num_json
    file_id = (abs_id % (num_local_shards * num_json)) % num_json
    link = link_format.format(global_id + 1, local_id, file_id)
    link_list.append(link)

with open("links.txt", "w") as f:
    for link in link_list:
        f.write(link + "\n")

assert len(link_list) == len(set(link_list)), "Links have duplication!"
print(f"After deduplication against previously selected shards: {len(link_list)}")
