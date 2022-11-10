import os

import torch
from miditok import MIDILike, REMI, Structured


def get_tokenizer_cls(data_args):
    cls = MIDILike
    if data_args.midi_tokenizer == 'REMI':
        cls = REMI
    elif data_args.midi_tokenizer == 'Structured':
        cls = Structured
    print(f'Use tokenizer {cls.__name__}')
    return cls


def load_embedding_model(data_args):
    tokenizer = get_tokenizer_cls(data_args)(sos_eos_tokens=True, mask=False)
    model = torch.nn.Embedding(len(tokenizer.vocab), data_args.in_channel)
    path_to_load = os.path.join(os.path.split(data_args.model_path)[0], "random_emb.torch")
    model.load_state_dict(torch.load(path_to_load))
    return model


def tokenize_e2e(args, indices):
    # v -> k
    tokenizer = get_tokenizer_cls(args)(sos_eos_tokens=True, mask=False)
    return [
        tokenizer.tokens_to_midi([[x[0].item() for x in seq]], [(0, False)])
        for seq in indices
    ]
