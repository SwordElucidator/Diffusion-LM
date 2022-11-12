import random

import torch
from miditok import MIDILike, REMI, Structured
from miditoolkit import MidiFile
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List


def advanced_remi_bar_block(tokens_list, image_size):
    block_length = image_size ** 2
    blocks = []
    for tokens in tokens_list:
        start_index = 0
        maximum = start_index + block_length - 1
        while maximum < len(tokens):
            # trace back
            if maximum + 1 == len(tokens) or tokens[maximum + 1] == 1:
                # 不用block了
                blocks.append(tokens[start_index: maximum + 1])
            else:
                while tokens[maximum] != 1:
                    maximum -= 1
                maximum -= 1
                blocks.append(tokens[start_index: maximum + 1] + [0] * (block_length - (maximum + 1 - start_index)))
            start_index = maximum + 1
            maximum = start_index + block_length - 1
    return blocks


def cut_midis():
    tokenizer = REMI(sos_eos_tokens=False, mask=False)
    tokens = tokenizer.midi_to_tokens(
        MidiFile('../datasets/midi/giant_midi_piano/train/Alkan, Charles-Valentin, Chapeau bas!, vFpL6KY-2W4.mid'))[0]
    if not os.path.isdir('experiment_advanced_padding'):
        os.mkdir('experiment_advanced_padding')
    image_size = 16
    for i in range(0, len(tokens) // (image_size ** 2) + 1):
        midi = tokenizer.tokens_to_midi([tokens[i * (image_size ** 2): (i + 1) * (image_size ** 2)]], [(0, False)])
        midi.dump(f"experiment_advanced_padding/{i}.mid")


if __name__ == '__main__':
    tokenizer = REMI(sos_eos_tokens=False, mask=False)
    tokens = tokenizer.midi_to_tokens(
        MidiFile('../datasets/midi/giant_midi_piano/train/Alkan, Charles-Valentin, Chapeau bas!, vFpL6KY-2W4.mid'))[0]
    image_size = 16
    blocks = advanced_remi_bar_block([tokens], image_size)
    for i, block in enumerate(blocks):
        midi = tokenizer.tokens_to_midi([block], [(0, False)])
        midi.dump(f"experiment_advanced_padding/{i}.mid")
