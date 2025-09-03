import os.path as osp
import math

f = open("links.txt")
all_links = f.readlines()
f.close()

num_links = len(all_links)
num_links_per_part = math.ceil(num_links / 10.)
tgt_dir = "/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yusu/datasets/dclm-600B-process/DCLM-600B-RAW"
f = open("processed_links.txt", 'w')

for i in range(num_links):
    link = all_links[i].strip()
    link_parts = link.split('/')
    file_name = '_'.join(link_parts[-3:])
    tgt = osp.join(tgt_dir, f'part_{i // num_links_per_part}', file_name)
    f.write(link + ' ' + tgt + '\n')

f.close()
