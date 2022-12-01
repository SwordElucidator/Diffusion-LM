import argparse
import os, json

import numpy as np
import torch as th

from symbolic_music.rounding import load_embedding_model
from transformers import set_seed
import torch.distributed as dist
from improved_diffusion.test_util import get_weights
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict, add_dict_to_argparser,
)


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


def prepare_args():
    set_seed(101)
    args = create_argparser().parse_args()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    args.noise_level = 0.0
    args.sigma_small = True

    dist_util.setup_dist()
    logger.configure()
    print(args.clip_denoised, 'clip_denoised')
    return args


def create_model(args):
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()
    return model, diffusion


def create_embedding(args, model):
    logger.log("load embedding models")
    print(os.path.split(args.model_path)[0])

    model_embs = load_embedding_model(args)
    model_embs.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_embs = model_embs.cuda() if th.cuda.is_available() else model_embs
    return get_weights(model_embs, args)


def save_results(args, samples, midi_list):
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


def get_score(input_embs, label_ids, model_control, t=None):
    label_ids2 = label_ids.clone()
    label_ids2[:, :65] = -100
    # print(label_ids2[:, 65:])
    # print(final.shape, tgt_embs.shape)
    # input_embs = th.cat([final, tgt_embs], dim=1)
    model_out = model_control(input_embs=input_embs,
                              labels=label_ids2, t=t)
    print(model_out.loss, 'final end')
    loss_fn = th.nn.CrossEntropyLoss(reduction='none')
    shifted_logits = model_out.logits[:, :-1].contiguous()
    shifted_labels = label_ids2[:, 1:].contiguous()
    loss = loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1)).reshape(
        shifted_labels.shape)
    return loss.sum(dim=-1).tolist()


def langevin_fn3(debug_lst, model_control, frozen_embedding_model, labels, step_size, sample, mean, sigma,
                 alpha, t, prev_sample):  # current best.
    if t[0].item() < 10:
        K = 0
    else:
        K = 3
    if t[0].item() > 0:
        tt = t[0].item() - 1
    else:
        tt = 200
    # label_ids = label_ids.cuda()
    # tgt_embs = frozen_embedding_model(label_ids[:, sample.size(1):])  # 只取了label的部分
    # tgt_embs = frozen_embedding_model(label_ids)

    # label_ids2 = label_ids.clone()
    input_embs_param = th.nn.Parameter(sample)
    # if False:
    #     input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
    #     debug_lst.append(get_score(input_embs, label_ids2, model_control, t=tt))
    with th.enable_grad():
        for i in range(K):
            optimizer = th.optim.Adagrad([input_embs_param], lr=step_size)
            optimizer.zero_grad()
            # input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
            # model_out = model_control(input_embs=input_embs,
            #                           labels=label_ids2, t=tt)
            # model_out = model_control(
            #     input_embs=input_embs_param, labels=label_ids2, t=tt
            # )
            import pdb
            pdb.set_trace()
            model_out = model_control(
                None, imput_embed=input_embs_param, labels=labels, timesteps=th.tensor([0] * sample.size(0))
            )

            coef = 0.01
            # coef=1.
            if sigma.mean() == 0:
                logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
            else:
                logp_term = coef * ((mean - input_embs_param) ** 2 / sigma).mean(dim=0).sum()
            # print(model_out.loss, f'start_{i}', logp_term.item(), t[0].item(), sigma.mean().item())
            loss = model_out.loss + logp_term
            loss.backward()
            optimizer.step()
            epsilon = th.randn_like(input_embs_param.data)
            input_embs_param = th.nn.Parameter((input_embs_param.data + 0.0 * sigma.mean().item() * epsilon).detach())
            # input_embs_param = th.nn.Parameter((input_embs_param.data +
            #                                    np.sqrt(2*sigma.mean().item()) * epsilon).detach())

    # input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
    # model_out = model_control(input_embs=input_embs,
    #                           labels=label_ids2,
    #                           t=tt)
    # print(model_out.loss, 'end')

    return input_embs_param.data

def langevin_fn4(debug_lst, model_control, model3, label_ids, step_size, sample, mean, sigma,
                 alpha, t, prev_sample): # current best.
    if t[0].item() < 10:
        K = 0
    else:
        K = 3

    if t[0].item() >0:
        tt =t[0].item() - 1
    else:
        tt = 200
    label_ids = label_ids.cuda()
    input_embs_param = th.nn.Parameter(sample)
    if False:
        input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
        debug_lst.append(get_score(input_embs, label_ids2, model_control, t=tt))
    with th.enable_grad():
        for i in range(K):
            optimizer = th.optim.Adagrad([input_embs_param], lr=step_size)
            optimizer.zero_grad()
            # print(input_embs_param.shape, label_ids.shape)
            model_out = model_control(input_embs=input_embs_param, pos_ids=label_ids, t=tt)

            coef = 0.0001 # prev default.
            # coef = 0.001
            # coef = 0.0005


            # coef=1.
            if sigma.mean() == 0:
                logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
            else:
                logp_term = coef * ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()
            print(model_out.loss, f'start_{i}', logp_term.item(),
                  t[0].item(), sigma.mean().item())
            loss = model_out.loss + logp_term
            loss.backward()
            optimizer.step()
            epsilon = th.randn_like(input_embs_param.data)
            input_embs_param = th.nn.Parameter((input_embs_param.data + 0.0*sigma.mean().item() * epsilon).detach())
            # input_embs_param = th.nn.Parameter((input_embs_param.data +
            #                                    np.sqrt(2*sigma.mean().item()) * epsilon).detach())

    model_out = model_control(input_embs=input_embs_param, pos_ids=label_ids, t=tt)
    print(model_out.loss, 'end')

    return input_embs_param.data

def langevin_fn_length(coeff, diffusion, partial_mask, diff_model, tgt_embs, step_size, sample, mean, sigma,
                 alpha, t, prev_sample): # current best.
    if t[0].item() < 10:
        K = 0
    else:
        K = 3

    if t[0].item() >0:
        tt =t[0].item() - 1
    else:
        tt = 200
    input_embs_param = th.nn.Parameter(sample)
    if False:
        input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
        debug_lst.append(get_score(input_embs, label_ids2, model_control, t=tt))
    with th.enable_grad():
        for i in range(K):
            optimizer = th.optim.Adagrad([input_embs_param], lr=step_size)
            optimizer.zero_grad()
            print(t.shape)
            # print(input_embs_param.shape, label_ids.shape)
            out = diffusion.p_mean_variance(
                diff_model,
                input_embs_param,
                t,
                clip_denoised=False,
                denoised_fn=None,
                model_kwargs={},
            )

            # model_out = model_control(input_embs=input_embs_param, pos_ids=label_ids, t=tt)
            coef = coeff
            # coef = 0.0001 # prev default.
            # coef = 0.001
            # coef = 0.0005


            # coef=1.
            if sigma.mean() == 0:
                logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
                infill_loss = (out['pred_xstart'][~partial_mask] - tgt_embs[~partial_mask]) ** 2
                infill_loss = infill_loss.mean(dim=0).sum()
            else:
                logp_term = coef * ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()
                # print(out['pred_xstart'].shape, tgt_embs.shape)
                # print(partial_mask[0])
                infill_loss = ((out['pred_xstart'][~partial_mask] - tgt_embs[~partial_mask]) ** 2).view(tgt_embs.size(0), -1, tgt_embs.size(-1) )
                # print(infill_loss.shape, ((mean - input_embs_param)**2).shape )
                infill_loss = (infill_loss/sigma.mean()).mean(dim=0).sum()
            print(infill_loss, f'start_{i}', logp_term.item(),
                  t[0].item(), sigma.mean().item())
            loss = logp_term + infill_loss
            loss.backward()
            optimizer.step()
            epsilon = th.randn_like(input_embs_param.data)
            input_embs_param = th.nn.Parameter((input_embs_param.data + 0.0*sigma.mean().item() * epsilon).detach())
            # input_embs_param = th.nn.Parameter((input_embs_param.data +
            #                                    np.sqrt(2*sigma.mean().item()) * epsilon).detach())

    # model_out = model_control(input_embs=input_embs_param, pos_ids=label_ids, t=tt)
    # print(model_out.loss, 'end')

    return input_embs_param.data

def langevin_fn_tree(coeff, model_control, label_ids, step_size, sample, mean, sigma,
                 alpha, t, prev_sample): # current best.
    if t[0].item() < 10:
        K = 0
    else:
        K = 3

    if t[0].item() >0:
        tt =t[0].item() - 1
    else:
        tt = 200
    label_ids = label_ids.cuda()
    input_embs_param = th.nn.Parameter(sample)

    with th.enable_grad():
        for i in range(K):
            optimizer = th.optim.Adagrad([input_embs_param], lr=step_size)
            optimizer.zero_grad()
            # print(input_embs_param.shape, label_ids.shape)
            model_out = model_control(input_embs=input_embs_param, parse_chart=label_ids, t=tt)

            # coef = 0.0001
            # coef = 0.001
            # coef = 0.01

            # coef = 0.1 # good for partial.
            # coef=0.001 # also good for full (more fluent).
            # coef=0.0001

            # coef=0.0005 # good for full.
            coef = coeff

            # coef = 0.5


            # coef=1.
            if sigma.mean() == 0:
                logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
            else:
                logp_term = coef * ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()
            # print(model_out.loss, f'start_{i}', logp_term.item(),
            #       t[0].item(), sigma.mean().item())
            loss = model_out.loss + logp_term
            loss.backward()
            optimizer.step()
            epsilon = th.randn_like(input_embs_param.data)
            input_embs_param = th.nn.Parameter((input_embs_param.data + 0.0*sigma.mean().item() * epsilon).detach())
            # input_embs_param = th.nn.Parameter((input_embs_param.data +
            #                                    np.sqrt(2*sigma.mean().item()) * epsilon).detach())

    # COMMENT OUT
    # model_out = model_control(input_embs=input_embs_param, parse_chart=label_ids, t=tt)
    # print(model_out.loss, 'end')

    return input_embs_param.data

def langevin_fn1(debug_lst, model_control, model3, label_ids, step_size, sample, mean, sigma,
                 alpha, t, prev_sample):  # current best.
    if t[0].item() < 10:
        K = 0
    else:
        K = 1
    # K = 3

    if t[0].item() > 0:
        tt = t[0].item() - 1
    else:
        tt = 200
    label_ids = label_ids.cuda()
    tgt_embs = model3(label_ids[:, sample.size(1):])

    label_ids2 = label_ids.clone()
    label_ids2[:, :65] = -100
    input_embs_param = th.nn.Parameter(sample)
    if True:
        input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
        debug_lst.append(get_score(input_embs, label_ids2, model_control, t=tt))
    with th.enable_grad():
        for i in range(K):
            optimizer = th.optim.Adagrad([input_embs_param], lr=step_size)
            optimizer.zero_grad()
            input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
            model_out = model_control(input_embs=input_embs,
                                      labels=label_ids2, t=tt)

            # coef = 0.0
            # if sigma.mean() == 0:
            #     logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
            # else:
            #     logp_term = coef * ((mean - input_embs_param) ** 2 / sigma).mean(dim=0).sum()
            print(model_out.loss, f'start_{i}', t[0].item(), sigma.mean().item())
            coef = 3.
            loss = model_out.loss # + logp_term
            loss.backward()
            # print(input_embs_param.grad.shape, )
            input_embs_param.data = input_embs_param.data - coef * sigma.mean().item() * input_embs_param.grad
            # optimizer.step()
            # epsilon = th.randn_like(input_embs_param.data)
            # input_embs_param = th.nn.Parameter((input_embs_param.data + 0.0 * sigma.mean().item() * epsilon).detach())
            # input_embs_param = th.nn.Parameter((input_embs_param.data +
            #                                    np.sqrt(2*sigma.mean().item()) * epsilon).detach())

    input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
    model_out = model_control(input_embs=input_embs,
                              labels=label_ids2,
                              t=tt)
    print(model_out.loss, 'end')
    # if True:
    #     debug_lst.append(get_score(input_embs, label_ids2, model_control, t=tt))

    return input_embs_param.data


def langevin_fn3_compose(debug_lst, model_control, model3, label_ids_lst, step_size, sample, mean, sigma,
                 alpha, t, prev_sample):  # current best.
    if t[0].item() < 10:
        K = 0
    else:
        K = 3
    # K = 3

    if t[0].item() > 0:
        tt = t[0].item() - 1
    else:
        tt = 200

    tgt_embs_lst = [model3(label_ids[:, sample.size(1):]) for label_ids in label_ids_lst]

    label_ids2_lst = []
    for label_ids in label_ids_lst:
        label_ids2 = label_ids.clone()
        label_ids2[:, :65] = -100
        label_ids2_lst.append(label_ids2)

    input_embs_param = th.nn.Parameter(sample)
    if True:
        part_score = []
        for (tgt_embs,label_ids2) in zip(tgt_embs_lst, label_ids2_lst):
            input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
            score_ = get_score(input_embs, label_ids2, model_control, t=tt)
            part_score.append(score_)
        debug_lst.append(part_score)
    with th.enable_grad():
        for i in range(K):
            optimizer = th.optim.Adagrad([input_embs_param], lr=step_size)
            optimizer.zero_grad()
            cum_loss = 0
            for (tgt_embs, label_ids2) in zip(tgt_embs_lst, label_ids2_lst):
                input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
                model_out = model_control(input_embs=input_embs,
                                          labels=label_ids2, t=tt)
                cum_loss += model_out.loss

            coef = 0.01
            if sigma.mean() == 0:
                logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
            else:
                logp_term = coef * ((mean - input_embs_param) ** 2 / sigma).mean(dim=0).sum()
            print(cum_loss, f'start_{i}', logp_term.item(), t[0].item(), sigma.mean().item())
            loss = cum_loss + logp_term
            loss.backward()
            optimizer.step()
            epsilon = th.randn_like(input_embs_param.data)
            input_embs_param = th.nn.Parameter((input_embs_param.data + 0.0 * sigma.mean().item() * epsilon).detach())

    part_score = []
    for (tgt_embs, label_ids2) in zip(tgt_embs_lst, label_ids2_lst):
        input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
        score_ = get_score(input_embs, label_ids2, model_control, t=tt)
        part_score.append(score_)

    return input_embs_param.data
