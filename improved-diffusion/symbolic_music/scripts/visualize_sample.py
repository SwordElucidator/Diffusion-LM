from miditok import REMI
from miditoolkit import MidiFile

if __name__ == '__main__':
    file = './../../genout_mono/diff_midi_midi_files_REMI_bar_block_rand32_music-transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_midi.model200000.pt.samples_1.0_1.mid'
    tokenizer = REMI(sos_eos_tokens=False, mask=False)
    tokens = tokenizer.midi_to_tokens(MidiFile(file))
    print(len(tokens[0]))
