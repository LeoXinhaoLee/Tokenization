'''
After generating links.txt
(1) Extend output file name with it's global and local shard id. Otherwise will overwrite each other.
awk '
{
    url = $0;
    sub("\\?download=true", "", url);
    split(url, parts, "/");
    filename = parts[length(parts)-2] "_" parts[length(parts)-1] "_" parts[length(parts)];
    print url, "/persistent_dclm/datasets/DCLM-200B-RAW/" filename;
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
sampled_abs_id = random.sample(file_abs_id, math.ceil(num_total_file / 20))  # 4T tokens -> 200B tokens
print(f'Sample shards number: {len(sampled_abs_id)}')

link_format = 'https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/resolve/main/global-shard_{:02}_of_10/local-shard_{}_of_10/shard_{:08}_processed.jsonl.zst?download=true'

link_list = []
for abs_id in sampled_abs_id:
    global_id = abs_id // (num_local_shards * num_json)
    local_id = (abs_id % (num_local_shards * num_json)) // num_json
    file_id = (abs_id % (num_local_shards * num_json)) % num_json
    link = link_format.format(global_id + 1, local_id, file_id)
    link_list.append(link)

with open("./tools/links.txt", "w") as f:
    for link in link_list:
        f.write(link + "\n")

assert len(link_list) == len(set(link_list)), "Links have duplication!"
print(f"After deduplication against previously selected shards: {len(link_list)}")
