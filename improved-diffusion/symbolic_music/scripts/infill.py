"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json, sys

import numpy as np
import torch as th

from transformers import set_seed
import torch.distributed as dist
from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer
from improved_diffusion.test_util import get_weights
from symbolic_music.rounding import load_embedding_model, tokens_list_to_midi_list, denoised_fn_round

from functools import partial
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

sys.path.insert(0, 'diffusion_lm/transformers/examples/pytorch/language-modeling')
from custom_trainer import Classifier_GPT2, Classifier_Times, Classifier_POS, Classifier_Tree
from infill_util import langevin_fn_tree


def __prepare_args():
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

    if args.eval_task_.startswith('control_'):
        args.diffusion_steps = 200  # DEBUG
    dist_util.setup_dist()
    logger.configure()
    print(args.clip_denoised, 'clip_denoised')
    return args


def __load_model(args):
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path))
    model.to(dist_util.dev())
    model.eval()
    return model, diffusion


def __load_embedding(args, model):
    logger.log("load embedding models")
    print(os.path.split(args.model_path)[0])
    model_embs = load_embedding_model(args)
    print('e2e, load the right model embeddings', '*' * 80)
    model_embs.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_embs = model_embs.cuda()
    frozen_embedding_model = get_weights(model_embs, args)
    return frozen_embedding_model


def main():
    set_seed(101)
    args = __prepare_args()
    model, diffusion = __load_model(args)
    frozen_embedding_model = __load_embedding(args, model)

    todo_pad_token = -1
    pad_token = 0  # FIXME

    right_pad = th.empty(64).fill_(pad_token).long()
    encoded_partial_seq = [th.cat([right_pad], dim=0)]

    # if args.eval_task_ == 'control_tree':
    model_control = Classifier_Tree.from_pretrained(
        'predictability/diff_models/e2e-tgt-tree_e=20_b=32_m=bert-base-uncased_'
        'wikitext-103-raw-v1_101_wp_full_multi16_cat').cuda()

    control_label_lst = []
    with open('diffusion_lm/improved-diffusion/control_gen/target_tree.json', 'r') as controlf:
        for line in controlf:
            control_label_lst.append(json.loads(line))
    control_constraints = []
    for label_class_dict in control_label_lst[100:]:
        padded_chart = th.LongTensor(label_class_dict['padded_chart'])
        label_ids = padded_chart
        langevin_fn_selected = partial(langevin_fn_tree, 0.0005, model_control,
                                       label_ids.expand(args.batch_size, -1, -1),
                                       0.1)
        control_constraints.append((langevin_fn_selected, [label_class_dict['tree']]))

    partial_seq = control_constraints
    encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
    print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-' * 20)

    logger.log("sampling...")

    for (encoded_seq, control_helper) in zip(encoded_partial_seq, partial_seq):
        all_images = []
        print(args.num_samples, encoded_seq.shape, 'encoded_seq.shape')
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            print(encoded_seq.shape)
            encoded_seq = encoded_seq.unsqueeze(0).expand(args.batch_size, -1)
            print(frozen_embedding_model.weight.device, encoded_seq.device)
            encoded_seq.masked_fill_(encoded_seq == todo_pad_token, 3)

            encoded_seq_hidden = frozen_embedding_model(encoded_seq.cuda())
            seqlen = encoded_seq.size(1)

            sample_shape = (args.batch_size, seqlen, args.in_channel,)

            langevin_fn_selected, label_class_attributes = control_helper
            print('-*' * 200, label_class_attributes, '-*' * 200)
            if args.use_ddim:
                loop_func_ = diffusion.ddim_sample_loop_progressive
            else:
                loop_func_ = diffusion.p_sample_loop_progressive

            for sample in loop_func_(
                    model,
                    sample_shape,
                    denoised_fn=partial(denoised_fn_round, frozen_embedding_model.cuda()),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    device=encoded_seq_hidden.device,
                    langevin_fn=langevin_fn_selected,
                    eta=args.eta,
            ):
                final = sample["sample"]

            if args.verbose == 'yes':
                label_ids = label_ids.expand(args.batch_size, -1, -1).cuda()

                model_out = model_control(input_embs=final,
                                          parse_chart=label_ids)
                print(model_out.loss, 'final end')
                loss_fn = th.nn.CrossEntropyLoss(reduction='none')
                shifted_logits = model_out.logits
                shifted_labels = label_ids
                loss = loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)),
                               shifted_labels.view(-1)).reshape(shifted_labels.shape)
                print(loss, loss.shape)
                word_lst = rounding_func(args.experiment, final, frozen_embedding_model, tokenizer)
                print(len(word_lst))
                for ww, ll in zip(word_lst, loss.sum(dim=-1).sum(dim=-1).tolist()):
                    print([ww], ll)

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

    word_lst_e2e = []
    print('decoding for e2e', )
    print(sample.shape)
    x_t = sample
    reshaped_x_t = x_t
    logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
    cands = th.topk(logits, k=1, dim=-1)
    tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
    for seq in cands.indices:
        tokens = " ".join([tokenizer[x[0].item()] for x in seq])
        word_lst_e2e.append(tokens)
    word_lst = word_lst_e2e

    out_path2 = os.path.join(args.out_dir,
                             f"{model_base_name}.infill_{args.eval_task_}_{shape_str}_{args.notes}.txt")
    fout = open(out_path2, 'w')
    for (xx) in zip(word_lst):
        print(xx[0], file=fout)
    fout.close()
    print(f'written the decoded output to {out_path2}')

    args.out_path2 = out_path2
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
    man_args = main()
    # eval(man_args)

