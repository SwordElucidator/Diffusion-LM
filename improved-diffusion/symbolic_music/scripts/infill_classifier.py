"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json, sys

import numpy as np
import torch as th

from symbolic_music.utils import get_tokenizer
from transformers import set_seed
import torch.distributed as dist
from improved_diffusion.rounding import rounding_func, load_tokenizer
from symbolic_music.rounding import denoised_fn_round

from functools import partial
from improved_diffusion import logger

sys.path.insert(0, 'diffusion_lm/transformers/examples/pytorch/language-modeling')
from custom_trainer import Classifier_GPT2, Classifier_Times, Classifier_POS, Classifier_Tree
from infill_util import langevin_fn_tree, prepare_args, create_model, create_embedding


def main():
    set_seed(101)
    args = prepare_args()
    model, diffusion = create_model(args)
    frozen_embedding_model = create_embedding(args, model)
    tokenizer = get_tokenizer(args)  # TODO

    todo_pad_token = -1
    pad_token = tokenizer['PAD_None']
    right_pad = th.empty(64).fill_(pad_token).long()
    encoded_partial_seq = [th.cat([right_pad], dim=0)]  # 1 * 64

    # 获取预训练的classifier
    model_control = Classifier_Tree.from_pretrained(
        'predictability/diff_models/e2e-tgt-tree_e=20_b=32_m=bert-base-uncased_'
        'wikitext-103-raw-v1_101_wp_full_multi16_cat').cuda()

    # get words
    control_label_lst = []
    with open('diffusion_lm/improved-diffusion/control_gen/target_tree.json', 'r') as controlf:
        for line in controlf:
            control_label_lst.append(json.loads(line))
    control_constraints = []
    for label_class_dict in control_label_lst[100:]:
        # (1, 64, 64)
        # [[0, 0, 0, ..., 0, 0, 0],
        #  [-100, 0, 43, ..., 0, 0, 0],
        #  [-100, -100, 0, ..., 0, 0, 0],
        #  ...,
        #  [-100, -100, -100, ..., 0, 0, 0],
        #  [-100, -100, -100, ..., -100, 0, 0],
        #  [-100, -100, -100, ..., -100, -100, 0]]
        label_ids = th.LongTensor(label_class_dict['padded_chart'])
        langevin_fn_selected = partial(
            langevin_fn_tree, 0.0005, model_control,
            label_ids.expand(args.batch_size, -1, -1),  # bsz,64, 64
            0.1
        )
        # example:
        # label_class_dict['tree'] = [ '(TOP (S (ADVP (NP (RB Only) (NNS feet)) (RB away) (PP (IN from) (NP (NNP Café) (NNP Sicilia)))) (, ,) (NP (DT The) (NNP Punter) (NN coffee) (NNP Shop)) (VP (VP (VBZ offers) (NP (NML (JJ low) (NN price)) (NN coffee))) (CC and) (VP (VBZ does) (RB not) (VP (VB have) (NP (NN family) (NNS restrooms))))) (. .) (. \n)))']
        control_constraints.append((langevin_fn_selected, [label_class_dict['tree']]))

    encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(control_constraints))]
    print(f'RUNNING FOR {len(control_constraints)} constraints.', '*-' * 20)

    logger.log("sampling...")

    for (encoded_seq, control_helper) in zip(encoded_partial_seq, control_constraints):
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

    if dist.get_rank() == 0:
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
                             f"{model_base_name}.infill_{args.eval_task_}_{args.notes}.txt")
    fout = open(out_path2, 'w')
    for (xx) in zip(word_lst):
        print(xx[0], file=fout)
    fout.close()
    print(f'written the decoded output to {out_path2}')

    args.out_path2 = out_path2
    return args


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

