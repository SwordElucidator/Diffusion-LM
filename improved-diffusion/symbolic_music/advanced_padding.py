from miditok import REMI
from miditoolkit import MidiFile


def advanced_remi_bar_block(tokens_list, block_size):
    blocks = []
    for tokens in tokens_list:
        start_index = 0
        maximum = start_index + block_size - 1
        while maximum < len(tokens):
            # trace back
            if maximum + 1 == len(tokens) or tokens[maximum + 1] == 1:
                # 不用block了
                blocks.append(tokens[start_index: maximum + 1])
            else:
                while tokens[maximum] != 1:
                    maximum -= 1
                maximum -= 1
                blocks.append(tokens[start_index: maximum + 1] + [0] * (block_size - (maximum + 1 - start_index)))
            start_index = maximum + 1
            maximum = start_index + block_size - 1
    return blocks


if __name__ == '__main__':
    tokenizer = REMI(sos_eos_tokens=False, mask=False)
    tokens = tokenizer.midi_to_tokens(
        MidiFile('../datasets/midi/giant_midi_piano/train/Alkan, Charles-Valentin, Chapeau bas!, vFpL6KY-2W4.mid'))[0]
    image_size = 16
    blocks = advanced_remi_bar_block([tokens], image_size ** 2)
    for block in blocks:
        print(len(block))
        print(block[0])
        print(block[-1])
    # for i, block in enumerate(blocks):
    #     midi = tokenizer.tokens_to_midi([block], [(0, False)])
    #     midi.dump(f"experiment_advanced_padding/{i}.mid")
