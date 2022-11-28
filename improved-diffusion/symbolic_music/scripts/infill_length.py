"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json, sys

import numpy as np
import torch as th

from symbolic_music.rounding import load_embedding_model, tokens_list_to_midi_list
from symbolic_music.utils import get_tokenizer
from transformers import set_seed
import torch.distributed as dist
from improved_diffusion.rounding import load_models, load_tokenizer
from improved_diffusion.test_util import get_weights, denoised_fn_round
from functools import partial
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def __save_results(args, samples, midi_list):
    # sample saving
    if dist.get_rank() == 0:
        model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
        out_path = os.path.join(args.out_dir, f"{model_base_name}.infill_{args.eval_task_}_{args.notes}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, samples)

    dist.barrier()

    if args.verbose == 'yes':
        # create midi files
        for i, midi in enumerate(midi_list):
            out_path2 = os.path.join(args.out_dir, f"{model_base_name}.infill_{args.eval_task_}_{args.notes}_{i}.mid")
            midi.dump(out_path2)


def main():
    set_seed(101)
    args = create_argparser().parse_args()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    args.__dict__.update(training_args)

    args.noise_level = 0.0
    args.sigma_small = True

    dist_util.setup_dist()
    logger.configure()
    print(args.clip_denoised, 'clip_denoised')
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    # model.load_state_dict(th.load(args.model_path))
    model.to(dist_util.dev())
    model.eval()

    logger.log("load embedding models")
    print(os.path.split(args.model_path)[0])

    model_embs = load_embedding_model(args)
    tokenizer = get_tokenizer(args)
    model_embs.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_embs = model_embs.cuda() if th.cuda.is_available() else model_embs
    model3 = get_weights(model_embs, args)


    # partial seq
    logger.log('load the partial sequences')

    file = './diffusion_models/diff_midi_midi_files_REMI_bar_block_rand32_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_midi/padded_tokens_list_valid.npz'
    arr = np.load(file)['arr_0']
    # partial_seq = arr[0]
    # partial_seq = ['A kid friendly venue named Alimentum is located on the riverside .',
    #                'Alimentum , situated by the river , is quite child friendly .']

    todo_pad_token = -1
    encoded_partial_seq = [th.LongTensor(arr[0][0: 30])]

    if args.eval_task_ == 'length':
        # 16 * 16 - 10
        right_length = args.image_size ** 2 - len(encoded_partial_seq[0])
        # fill the left part
        right_pad = th.empty(right_length).fill_(todo_pad_token).long()
        # seq + right padding
        encoded_partial_seq = [th.cat([seq, right_pad], dim=0) for seq in encoded_partial_seq]
        # put the tgt_len token-> end
        # encoded_partial_seq[0][args.tgt_len - 1] = tokens2id['END']
        # put tgt_len -> start
        encoded_partial_seq[0] = th.cat([
            th.tensor(encoded_partial_seq[0][:args.tgt_len]),
            th.tensor(tokenizer.vocab['PAD_None'] * args.image_size ** 2 - args.tgt_len)
        ])
    print(encoded_partial_seq[0], len(encoded_partial_seq[0]))

    logger.log("sampling...")
    for encoded_seq in encoded_partial_seq:
        all_images = []
        print(args.num_samples, encoded_seq.shape, 'encoded_seq.shape')
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            print(encoded_seq.shape)
            encoded_seq = encoded_seq.unsqueeze(0).expand(args.batch_size, -1)
            print(model_embs.weight.device, encoded_seq.device)
            partial_mask_temp = (encoded_seq == todo_pad_token).view(args.batch_size, -1)
            encoded_seq.masked_fill_(encoded_seq == todo_pad_token, 3)

            encoded_seq_hidden = model_embs(encoded_seq.cuda() if th.cuda.is_available() else encoded_seq)
            seqlen = encoded_seq.size(1)
            partial_mask = partial_mask_temp.unsqueeze(-1).expand(-1, -1, args.in_channel)
            sample_shape = (args.batch_size, seqlen, args.in_channel,)
            loop_func_ = diffusion.p_sample_loop_progressive_infill

            for sample in loop_func_(
                    model,
                    sample_shape,
                    encoded_seq_hidden,
                    partial_mask,
                    denoised_fn=partial(denoised_fn_round, args, model3.cuda() if th.cuda.is_available() else model3),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    device=encoded_seq_hidden.device,
                    greedy=False,
            ):
                final = sample["sample"]

            sample = final
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            logger.log(f"created {len(all_images) * args.batch_size} samples")

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'

    dist.barrier()
    logger.log("sampling complete")

    if args.verbose == 'yes':
        word_lst_e2e = []
        print('decoding for e2e', )
        print(sample.shape)
        x_t = sample
        reshaped_x_t = x_t
        logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1)
        __save_results(args, sample, tokens_list_to_midi_list(args, cands.indices))

    # args.out_path2 = out_path2
    return args


def create_argparser():
    defaults = dict(
        data_dir="", clip_denoised=False, use_ddim=False, eta=1.0, num_samples=50, batch_size=1, model_path="",
        out_dir="diffusion_lm/improved_diffusion/out_gen",
        emb_scale_factor=1.0, split='train', debug_path='', eval_task_='infill',
        partial_seq="", partial_seq_file="", verbose='yes', tgt_len=15, t_merge=200, interp_coef=0.5, notes='',
        start_idx=0, end_idx=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def eval(args):
    if args.modality == 'e2e-tgt':
        model_name_path = "predictability/diff_models/e2e-tgt_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_None"

        COMMAND = f"python scripts/ppl_under_ar.py " \
                  f"--model_path {args.model_path} " \
                  f"--modality {args.modality}  --experiment random " \
                  f"--model_name_or_path {model_name_path} " \
                  f"--input_text {args.out_path2}  --mode eval"
        print(COMMAND)
        os.system(COMMAND)


if __name__ == "__main__":
    args = main()
    # import numpy as np

    # if args.verbose != 'pipe':
    #     eval(args)

