import torch
from sklearn.decomposition import PCA
from miditok import REMI


def pca(path):
    word_embedding = torch.load(path)['word_embedding.weight']
    tokenizer = REMI(sos_eos_tokens=False, mask=False)
    three_dim = PCA(random_state=0).fit_transform(word_embedding)[:, :3]
    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    pca('./diffusion_models/diff_midi_giant_midi_piano_REMI_bar_block_rand32_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_midi/model400000.pt')
