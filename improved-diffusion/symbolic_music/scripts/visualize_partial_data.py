import numpy as np
from miditok import REMI

if __name__ == '__main__':
    file = './../../diffusion_models/diff_midi_midi_files_REMI_bar_block_rand32_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_midi/padded_tokens_list_valid.npz'
    arr = np.load(file)['arr_0']
    tokenizer = REMI(sos_eos_tokens=False, mask=False)
    midi = tokenizer.tokens_to_midi([arr[531][0: 21]], [(0, False)])
    midi.dump(f'test.mid')
