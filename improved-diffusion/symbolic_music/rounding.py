import os

import numpy as np
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


def tokens_list_to_midi_list(args, indices):
    # v -> k
    tokenizer = get_tokenizer(args)
    return [
        tokenizer.tokens_to_midi([[x[0].item() for x in seq]], [(0, False)])
        for seq in indices
    ]


def denoised_fn_round(model, text_emb, t):
    down_proj_emb = model.weight  # input_embs
    old_shape = text_emb.shape
    old_device = text_emb.device

    def get_efficient_knn(down_proj_emb, text_emb, dist='l2'):
        if dist == 'l2':
            emb_norm = (down_proj_emb ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            # print(emb_norm.shape, arr_norm.shape)
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(down_proj_emb, text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            dist = torch.clamp(dist, 0.0, np.inf)
        topk_out = torch.topk(-dist, k=1, dim=0)
        return topk_out.values, topk_out.indices

    dist = 'l2'
    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))
    val, indices = get_efficient_knn(down_proj_emb, text_emb.to(down_proj_emb.device), dist=dist)
    rounded_tokens = indices[0]
    new_embeds = model(rounded_tokens).view(old_shape).to(old_device)
    return new_embeds
