import numpy as np
from miditok import REMI

if __name__ == '__main__':
    file = './../../diffusion_models/diff_midi_midi_files_REMI_bar_block_rand64_music-transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_midi/padded_tokens_list_train.npz'
    arr = np.load(file)['arr_0']
    indexes = np.random.choice(len(arr), 10, replace=False)
    tokenizer = REMI(sos_eos_tokens=False, mask=False)
    for index in indexes:
        midi = tokenizer.tokens_to_midi([arr[index]], [(0, False)])
        midi.dump(f'{index}.mid')
