import pdb

# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
from itertools import chain
from pathlib import Path
import pickle
import subprocess
import mmap
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from pytorch_lightning import LightningDataModule


class LMDataModule(LightningDataModule):
    def __init__(self, dataset_name, tokenizer_name, dataset_config_name=None,
                 cache_dir=None, val_ratio=0.0005, val_split_seed=2357, add_eos=True,
                 detokenize=False, val_only=False, num_workers=1,
                 raw_json_path=None, finetune_ratio=None, pad_to_multiple_of=0):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.num_workers = num_workers

        # @xinhao: add option to specify raw json path directly
        self.raw_json_path = raw_json_path
        self.finetune_ratio = finetune_ratio

        # @xinhao: add option for end-document padding
        self.pad_to_multiple_of = pad_to_multiple_of

    def prepare_data(self):
        if self.cache_dir is None:
            # Just download the dataset
            load_dataset(self.dataset_name, self.dataset_config_name)
        else:
            # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == 'test' and hasattr(self, 'dataset_test'):
            return
        concat_ids, self.tokenizer = self.process_dataset()
        self.vocab_size = len(self.tokenizer)

    def process_dataset(self):
        cache_dir = None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        if self.raw_json_path is not None:
            raw_datasets = load_dataset('json', data_files=self.raw_json_path)
        else:
            raw_datasets = load_dataset(self.dataset_name, self.dataset_config_name)

        # https://github.com/stanford-crfm/mistral/blob/main/src/corpora/auto.py
        if 'validation' not in raw_datasets:
            assert "train" in raw_datasets, "You must have train in raw_datasets to make a validation raw_datasets"
            raw_datasets = raw_datasets["train"].train_test_split(
                test_size=self.val_ratio,
                seed=self.val_split_seed,
                shuffle=True  # Otherwise test will be at the end of the dataset
            )
            raw_datasets['validation'] = raw_datasets['test']
            if 'books3_splitted_finetune' in str(self.cache_dir):
                assert self.finetune_ratio is not None, "Must specify a finetune data ratio!"
                del raw_datasets['test']
                validation = raw_datasets['validation']
                raw_datasets = raw_datasets["train"].train_test_split(
                    test_size=self.finetune_ratio,
                    seed=self.val_split_seed,
                    shuffle=True  # split train set into pre-train and finetune
                )
                raw_datasets['finetune'] = raw_datasets['test']
                raw_datasets['validation'] = validation
                raw_datasets['test'] = validation

        if self.val_only:  # Should only be used for evaluation, not for training
            raw_datasets['train'] = raw_datasets['validation']

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        if self.add_eos:
            add_eos = lambda seq: (seq + tokenizer.eos_token) if seq else seq
            add_eos_batched = lambda seqs: [add_eos(seq) for seq in seqs]
            tokenize = lambda example: tokenizer(add_eos_batched(example[text_column_name]))
        else:
            tokenize = lambda example: tokenizer(example[text_column_name])

        # @xinhao: integrate Jiarui's script
        if self.pad_to_multiple_of > 0:
            _tokenize = tokenize
            def pad_to_multiple(tokens, pad_token_id, multiple):
                length = len(tokens)
                padding_length = (multiple - length % multiple) % multiple
                return tokens + [pad_token_id] * padding_length
            def tokenize_pad_to_multiple(example):
                tokenize_dic = _tokenize(example)
                input_ids = tokenize_dic.pop('input_ids')
                input_ids = [pad_to_multiple(tokens, tokenizer.bos_token_id, self.pad_to_multiple_of) for tokens in input_ids]
                return {'input_ids': input_ids, **tokenize_dic}
            tokenize = tokenize_pad_to_multiple

        dtype = np.uint16 if tokenizer.vocab_size < 64 * 1024 else np.int32
        def tokenize_concat(examples):
            # We just need 'input_ids', not 'attention_mask' (since it's all 1)
            input_ids = np.fromiter(chain(*tokenize(examples)['input_ids']), dtype=dtype)
            # Need to return a list since we're doing batched processing
            return {'input_ids': [input_ids], 'len': [len(input_ids)]}
        tokenized_datasets = raw_datasets.map(
            tokenize_concat,
            batched=True,
            num_proc=max(self.num_workers, 1),
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

        # Use disk
        concat_ids = {}
        assert cache_dir is not None
        cache_dir.mkdir(parents=True, exist_ok=True)

        def write_ids_to_disk(example, filename):
            with open(filename, 'r+b') as f:
                mm = mmap.mmap(f.fileno(), 0)
                start_idx = example['len_offset'] - len(example['input_ids'])
                array_len = len(example['input_ids'])
                arr = np.ndarray((array_len,), dtype=dtype, buffer=mm, offset=np.dtype(dtype).itemsize * start_idx)
                arr[:] = example['input_ids']
                mm.flush()

        for name, ds in tokenized_datasets.items():
            tokenized_datasets[name] = ds.add_column('len_offset', np.cumsum(ds['len']))
            array_len = tokenized_datasets[name][-1]['len_offset']
            filename = cache_dir / f'{name}.bin'

            # Need to create the file with this specific size first
            # https://ostechnix.com/create-files-certain-size-linux/
            subprocess.run(['truncate', '-s', str(array_len * np.dtype(dtype).itemsize), str(filename)], check=True)

            tokenized_datasets[name].map(
                write_ids_to_disk,
                fn_kwargs={'filename': filename},
                batched=False,
                num_proc=max(self.num_workers, 1),
                desc="Concatenating examples",
            )
            concat_ids[name] = np.memmap(filename, dtype=dtype, mode='r', shape=(array_len,))

        if cache_dir is not None:
            self._save_to_cache(concat_ids, tokenizer, cache_dir)
            for name in concat_ids:
                Path(cache_dir / f'{name}.bin').unlink()
        return concat_ids, tokenizer

    def _save_to_cache(self, concat_ids, tokenizer, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f'Saving to cache at {str(cache_dir)}')
        for k, v in concat_ids.items():
            np.save(cache_dir / f'{k}.npy', v)
        with open(cache_dir / 'tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        # logger.info(f'Load from cache at {str(cache_dir)}')
        print(f'Load from cache at {str(cache_dir)}')
        if self.dataset_name == 'books3_splitted_finetune':
            concat_ids = {split: np.load(cache_dir / f'{split}.npy', mmap_mode='r')
                          for split in ['train', 'finetune', 'validation', 'test']}
        else:
            concat_ids = {split: np.load(cache_dir / f'{split}.npy', mmap_mode='r')
                          for split in ['train', 'validation', 'test']}
        with open(cache_dir / 'tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return concat_ids, tokenizer

    @property
    def _cache_dir_name(self):
        return f'tokenizer_name-{self.tokenizer_name}-val_ratio-{self.val_ratio}-val_split_seed-{self.val_split_seed}-add_eos-{self.add_eos}-detokenize-{self.detokenize}'
