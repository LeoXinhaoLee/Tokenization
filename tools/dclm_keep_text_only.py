# /// script
# dependencies = [
#     "zstandard>=0.24.0",
#     "tqdm>=4.67.1"
# ]
# ///
import os
import glob
import pdb
from pathlib import Path
import zstandard as zstd
import json
from tqdm import tqdm
import argparse

def main(args):
    src_dir = Path(args.src_dir)
    tgt_dir = Path(args.tgt_dir)
    tgt_dir.mkdir(parents=True, exist_ok=True)
    json_list = list(src_dir.glob("*.jsonl.zst"))

    for file in tqdm(json_list):
        with open(file, 'rb') as compressed_file:
            dctx = zstd.ZstdDecompressor()

            with dctx.stream_reader(compressed_file) as reader:
                decompressed_data = reader.read().decode('utf-8')
                lines = decompressed_data.splitlines()

                tgt_file = tgt_dir / file.with_suffix('').name  # src/a.jsonl.zst -> tgt/a.jsonl
                with open(tgt_file, 'w') as out_file:
                    for line in lines:
                        record = json.loads(line)
                        new_record = dict(text=record['text'])
                        out_file.write(json.dumps(new_record) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default="dclm")
    parser.add_argument('--tgt_dir', type=str, default="dclm-text")
    args = parser.parse_args()

    main(args)

