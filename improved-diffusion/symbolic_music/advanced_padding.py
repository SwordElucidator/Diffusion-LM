import os

import numpy as np
from miditok import REMI
from miditoolkit import MidiFile


def advanced_remi_bar_block(tokens_list, block_size, skip_paddings_ratio=0.2):
    blocks = []
    skipped = 0
    for i, tokens in enumerate(tokens_list):
        start_index = 0
        maximum = start_index + block_size - 1
        while maximum < len(tokens):
            assert tokens[start_index] == 1
            # 跳过repeated 1
            should_break = False
            while tokens[start_index + 1] == 1:
                start_index += 1
                maximum += 1
                if maximum >= len(tokens):
                    should_break = True
                    break
            if should_break:
                break
            # print(start_index)
            # trace back
            if maximum + 1 == len(tokens) or tokens[maximum + 1] == 1:
                # 不用block了
                blocks.append(tokens[start_index: maximum + 1])
            else:
                while tokens[maximum] != 1:
                    maximum -= 1
                maximum -= 1
                if start_index > maximum:
                    # 小节太鬼畜了，直接跳过
                    start_index += 1
                    while tokens[start_index] != 1:
                        start_index += 1
                        if start_index == len(tokens):
                            break
                    maximum = start_index + block_size - 1
                    continue
                if block_size - (maximum + 1 - start_index) <= block_size * skip_paddings_ratio:
                    blocks.append(tokens[start_index: maximum + 1] + [0] * (block_size - (maximum + 1 - start_index)))
                else:
                    skipped += 1
            start_index = maximum + 1
            maximum = start_index + block_size - 1
    print('%s blocks skipped' % skipped)
    return blocks


if __name__ == '__main__':
    tokenizer = REMI(sos_eos_tokens=False, mask=False)
    image_size = 16
    i = 0
    item = 'train'
    out = []
    print(f"working on {item}")
    for midi_file_name in os.listdir(f'../datasets/midi/giant_midi_piano/{item}/'):
        if midi_file_name.endswith('.mid'):
            tokens_list = tokenizer.midi_to_tokens(
                MidiFile(os.path.join(f'../datasets/midi/giant_midi_piano/{item}/', midi_file_name)))
            out.extend(advanced_remi_bar_block(tokens_list, image_size ** 2))
    print(len(out))
    np.savez(f'padded_tokens_list_{item}.npz', out)
    #
    # tokens = tokenizer.midi_to_tokens(
    #     MidiFile('../datasets/midi/giant_midi_piano/train/Alkan, Charles-Valentin, Chapeau bas!, vFpL6KY-2W4.mid'))[0]
    #
    # blocks = advanced_remi_bar_block([tokens], image_size ** 2)
    # for block in blocks:
    #     print(len(block))
    #     print(block[0])
    #     print(block[-1])
    # for i, block in enumerate(blocks):
    #     midi = tokenizer.tokens_to_midi([block], [(0, False)])
    #     midi.dump(f"experiment_advanced_padding/{i}.mid")
