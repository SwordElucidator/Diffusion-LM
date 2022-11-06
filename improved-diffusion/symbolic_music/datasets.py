import random

import torch
from miditok import MIDILike
from miditoolkit import MidiFile
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List


def __create_embedding_model(data_args, vocab_size):
    model = torch.nn.Embedding(vocab_size, data_args.in_channel)  # in_channel: embedding dim
    print('initializing the random embeddings', model)
    torch.nn.init.normal_(model.weight)
    path_save = f'{data_args.checkpoint_path}/random_emb.torch'
    print(f'save the random encoder to {data_args.checkpoint_path}/random_emb.torch')
    torch.save(model.state_dict(), path_save)
    return model


def __padding(data_args, tokens_list, block_size) -> List[List[int]]:
    """
    block padding, will make blocks for long examples. note that [s] and [end] might be lost in many many tracks
    """
    if data_args.padding_mode == 'block':
        print('using block padding')
        concatenated_tokens = sum(tokens_list, [])
        total_length = (len(concatenated_tokens) // block_size) * block_size
        print(f'total length: {total_length}')
        return [concatenated_tokens[i: i + block_size] for i in range(0, total_length, block_size)]
    if data_args.padding_mode == 'pad_and_truncate':
        print('using pad & truncate padding')
        return [
            tokens[0: block_size] + [0] * max(0, block_size - len(tokens)) for tokens in tokens_list
        ]
    raise NotImplementedError


class MidiDataset(Dataset):
    def __init__(
            self, midi_data_list, resolution, data_args, model_arch, eigen_transform=None,
            mapping_func=None, model_emb=None
    ):
        super().__init__()
        self.resolution = resolution
        self.midi_data_list = midi_data_list
        self.length = len(self.midi_data_list)
        self.model_arch = model_arch
        self.data_args = data_args
        print(self.resolution)
        self.eigen_transform = eigen_transform
        self.mapping_func = mapping_func
        self.model_emb = model_emb

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        arr = np.array(self.midi_data_list[idx]['hidden_states'], dtype=np.float32)
        if self.eigen_transform is not None:
            old_shape = arr.shape
            # arr = arr.reshape(1, -1) @ self.eigen_transform
            arr = arr.reshape(1, -1) - self.eigen_transform['mean']
            arr = arr @ self.eigen_transform['map']
            arr = arr.reshape(old_shape)

        if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
            # print(arr.dtype)
            # print(self.data_args.noise_level, 'using the noise level.')
            arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)
            # print(arr.dtype)

        out_dict = {'input_ids': np.array(self.midi_data_list[idx]['input_ids'])}
        if self.data_args.experiment_mode == 'conditional_gen':  # TODO not implementing conditional gen for now
            out_dict['src_ids'] = np.array(self.midi_data_list[idx]['src_ids'])
            out_dict['src_mask'] = np.array(self.midi_data_list[idx]['src_mask'])
        return arr, out_dict


def create_midi_dataloader(
        *, batch_size, data_args=None, split='train', embedding_model=None, dataset_partition=1
):
    """
    lower the complexity for now.
    Will add more experiments later
    """
    print("Creating midi dataloader...")
    tokenizer = MIDILike(sos_eos_tokens=True, mask=False)  # it's ok to always add a mask token
    # data_args.data_path
    tokens_list = []
    print(f"Start tokenize files in {os.path.join(data_args.data_path, split)} with partition={dataset_partition}")
    for midi_file_name in os.listdir(os.path.join(data_args.data_path, split)):
        if random.random() > dataset_partition:
            continue
        if midi_file_name.endswith('.mid'):
            # will have a very long size for each
            tokens = tokenizer.midi_to_tokens(MidiFile(os.path.join(data_args.data_path, split, midi_file_name)))
            tokens_list.append(tokens[0])
    print(f'Finish tokenize {len(tokens_list)} items')
    if not embedding_model:
        embedding_model = __create_embedding_model(data_args, vocab_size=len(tokenizer.vocab))
    print(f"Start padding...")
    padded_tokens_list = __padding(data_args, tokens_list, data_args.image_size ** 2)
    data_list = [
        {
            'input_ids': padded_tokens, 
            'hidden_states': embedding_model(torch.tensor(padded_tokens)).cpu().tolist()
        }
        for padded_tokens in padded_tokens_list
    ]

    dataset = MidiDataset(
        data_list,
        data_args.image_size,
        data_args,
        model_arch=data_args.model_arch,  # transformer for NLP / MIDI, or probably use better music transformer? TODO
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,  # 64,
        drop_last=True,
        shuffle=True,
        num_workers=1,
    )
    while True:
        yield from data_loader
