import argparse
import json
import os

import numpy as np
import torch as th
from improved_diffusion import dist_util
from improved_diffusion.script_util import create_model_and_diffusion, args_to_dict, model_and_diffusion_defaults, \
    add_dict_to_argparser
from symbolic_music.rounding import tokenize_e2e
from transformers import set_seed
from improved_diffusion import dist_util, logger


def main():
    set_seed(101)
    args = create_argparser().parse_args()
    print('Start with args:')
    print(args.__dict__)

    dist_util.setup_dist()
    logger.configure()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)
    args.sigma_small = True

    arr = np.load(args.npz_path)['arr_0']

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    reshaped_x_t = th.tensor(arr)
    logits = model.get_logits(reshaped_x_t)
    cands = th.topk(logits, k=1, dim=-1)
    print(f"cands is {cands}")
    word_lst_e2e = tokenize_e2e(args, cands.indices)
    import pdb
    pdb.set_trace()



def create_argparser():
    defaults = dict(
        npz_path=None,
        clip_denoised=False,
        num_samples=50,#10000,
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
                         emb_scale_factor=1.0, top_p=-1., split='valid', clamp='clamp',
                         midi_tokenizer='REMI')
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    # defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
