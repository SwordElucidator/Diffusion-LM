import torch
from miditok import MIDILike


def load_embedding_model(data_args):
    tokenizer = MIDILike(sos_eos_tokens=True, mask=False)
    model = torch.nn.Embedding(len(tokenizer.vocab), data_args.in_channel)
    path_to_load = f'{data_args.checkpoint_path}/random_emb.torch'
    model.load_state_dict(torch.load(path_to_load))
    return model
