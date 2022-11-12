import os

import torch
from symbolic_music.utils import get_tokenizer


def load_embedding_model(data_args):
    tokenizer = get_tokenizer(data_args)
    model = torch.nn.Embedding(len(tokenizer.vocab), data_args.in_channel)
    if hasattr(data_args, 'model_path'):
        path_to_load = os.path.join(os.path.split(data_args.model_path)[0], "random_emb.torch")
    else:
        path_to_load = f'{data_args.checkpoint_path}/random_emb.torch'
    model.load_state_dict(torch.load(path_to_load))
    return model


def tokenize_e2e(args, indices):
    # v -> k
    tokenizer = get_tokenizer(args)
    return [
        tokenizer.tokens_to_midi([[x[0].item() for x in seq]], [(0, False)])
        for seq in indices
    ]
