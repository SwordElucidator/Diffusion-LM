"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
import time

import numpy as np
import torch
import torch as th
import torch.distributed as dist

from symbolic_music.rounding import load_embedding_model, tokens_list_to_midi_list, denoised_fn_round
from transformers import set_seed
from improved_diffusion.test_util import get_weights
from improved_diffusion import dist_util, logger
from functools import partial
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def __prepare_args():
    args = create_argparser().parse_args()
    print('Start with args:')
    print(args.__dict__)
    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)
    args.sigma_small = True
    return args


def __prepare_models(args):
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'the parameter count is {pytorch_total_params}')

    print(diffusion.rescale_timesteps, 'a marker for whether we are in the debug mode')
    model.to(dist_util.dev())
    model.eval()  # DEBUG
    return model, diffusion


def __prepare_embedding_model(args, model):
    embedding_model = load_embedding_model(args)  # load the embedding model
    print('e2e, load the right model embeddings', '*' * 80)
    # set embedding weight to learnt ones
    embedding_model.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    frozen_embedding_model = get_weights(embedding_model, args)
    return frozen_embedding_model


def __sampling(args, model, diffusion, frozen_embedding_model):
    all_images = []
    print(args.num_samples)

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.experiment_mode == 'conditional_gen':
            pass  # TODO condition
        sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
        if args.mbr_sample > 1 and args.experiment_mode == 'conditional_gen':
            sample_shape = (args.batch_size * args.mbr_sample, args.image_size ** 2, args.in_channel)
        else:
            sample_shape = (args.batch_size, args.image_size ** 2, args.in_channel)
        print(sample_shape)
        # make a batch of sample
        sample = sample_fn(
            model,
            sample_shape,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(
                denoised_fn_round,
                frozen_embedding_model.cuda() if torch.cuda.is_available() else frozen_embedding_model
            ) if args.clamp == 'clamp' else None,
            model_kwargs=model_kwargs,
            top_p=args.top_p,
        )
        print(sample.shape)
        # collect results from multi processes
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    # make flat
    arr = np.concatenate(all_images, axis=0)
    print(arr.shape, 'full shape')
    return arr[: args.num_samples * args.mbr_sample]


def __save_results(args, samples, midi_list):
    # sample saving
    if dist.get_rank() == 0:
        model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
        out_path = os.path.join(args.out_dir, f"{model_base_name}.samples_{args.top_p}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, samples)

    dist.barrier()

    if args.verbose == 'yes':
        # create midi files
        for i, midi in enumerate(midi_list):
            out_path2 = os.path.join(args.out_dir, f"{model_base_name}.samples_{args.top_p}_{i}.mid")
            midi.dump(out_path2)


def __calc_indices(samples, model):
    # (sample_size, image_size **2, embedding)
    x_t = th.tensor(samples).cuda() if torch.cuda.is_available() else th.tensor(samples)  # for debug
    # go over the lm head and get logits (sample_size, image_size **2, vocab_len)
    logits = model.get_logits(x_t)  # bsz, seqlen, vocab
    cands = th.topk(logits, k=1, dim=-1)
    print(f"cands is {cands}")
    return cands.indices


def main():  # !!! don't use checkpoint_path from hyper
    set_seed(101)
    dist_util.setup_dist()
    logger.configure()

    args = __prepare_args()
    model, diffusion = __prepare_models(args)

    if args.experiment_mode == 'conditional_gen':
        pass  # TODO

    frozen_embedding_model = __prepare_embedding_model(args, model)
    logger.log("sampling...")
    start = time.time()

    samples = __sampling(args, model, diffusion, frozen_embedding_model)
    print(samples.shape)
    logger.log("sampling complete")
    print(f'Sample cost time: {time.time() - start}')

    midi_list = tokens_list_to_midi_list(args, __calc_indices(samples, model))
    __save_results(args, samples, midi_list)


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=50,
        batch_size=64,
        use_ddim=False,
        mbr_sample=1,
        model_path="",
        model_arch='conv-unet',
        verbose='yes',
        out_dir="diffusion_lm/improved_diffusion/out_gen"
    )
    text_defaults = dict(modality='text',
                         dataset_name='wikitext',
                         dataset_config_name='wikitext-2-raw-v1',
                         model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
                         experiment='gpt2_pre_compress', model_arch='trans-unet',
                         preprocessing_num_workers=1,
                         emb_scale_factor=1.0, top_p=-1., split='valid', clamp='clamp', midi_tokenizer='REMI')
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
