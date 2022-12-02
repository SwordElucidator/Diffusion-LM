"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import os, json, sys
import torch as th
from music_classifier.transfomer_net import TransformerNetClassifierModel
from symbolic_music.rounding import tokens_list_to_midi_list
from symbolic_music.utils import get_tokenizer
from transformers import set_seed, BertConfig
import torch.distributed as dist
from improved_diffusion.test_util import denoised_fn_round
from functools import partial
from improved_diffusion import logger
from infill_util import langevin_fn3, prepare_args, create_model, create_embedding, save_results


def main():
    set_seed(101)
    args = prepare_args()
    model, diffusion = create_model(args)
    frozen_embedding_model = create_embedding(args, model)
    tokenizer = get_tokenizer(args)  # TODO
    
    logger.log('load the partial sequences')

    pad_token = tokenizer.vocab['PAD_None']

    # length = 64
    right_pad = th.empty(args.image_size ** 2).fill_(pad_token).long()
    encoded_partial_seqs = [th.cat([right_pad], dim=0)]
    # encoded_partial_seqs[0][0] = tokens2id['START']
    # encoded_partial_seqs[0][args.tgt_len] = tokens2id['END']

    if args.eval_task_ == 'control_attribute':
        config = BertConfig.from_json_file(os.path.join('./classifier_models/bert/bert-config.json'))
        model_control = TransformerNetClassifierModel(config, args.in_channel)
        model_control.load_state_dict(th.load('./classifier_models/bert/checkpoint-30000/pytorch_model.bin', map_location=th.device('cpu')))
        learned_embeddings = th.load(args.model_path, map_location=th.device('cpu'))['word_embedding.weight']
        model_control.transformer_net.word_embedding.weight.data = learned_embeddings.clone()
        model_control.transformer_net.word_embedding.weight.requires_grad = False

        control_label_lst = [config.label2id["0"], config.label2id["42"], config.label2id["52", config.label2id["70"]]]   # TODO
        control_constraints = []
        for label in control_label_lst:
            # label = [-100] * 64 + [tokens2id.get(x, tokens2id['UNK']) for x in label_class]
            debug_lst = []

            langevin_fn_selected = partial(
                langevin_fn3, debug_lst, model_control.cuda() if th.cuda.is_available() else model_control,
                frozen_embedding_model.cuda() if th.cuda.is_available() else frozen_embedding_model,
                # th.tensor([label]).expand(args.batch_size, -1),
                th.tensor([label]).expand(args.batch_size).cuda() if th.cuda.is_available() else th.tensor([label]).expand(args.batch_size),
                # label_ids.expand(args.batch_size, -1),  # [batch_size, label_length]
                0.1
            )
            control_constraints.append((langevin_fn_selected, label))

        # [constraint count, img ** 2]  对着每个constraint，做一个seq
        encoded_partial_seqs = [encoded_partial_seqs[0] for _ in range(len(control_constraints))]
        assert len(control_constraints) == len(encoded_partial_seqs)
        print(f'RUNNING FOR {len(control_constraints)} constraints.', '*-' * 20)

    print(encoded_partial_seqs[0], len(encoded_partial_seqs[0]))

    logger.log("sampling...")
    for (encoded_seq, (langevin_fn_selected, label)) in zip(encoded_partial_seqs, control_constraints):
        all_images = []
        print(args.num_samples, encoded_seq.shape, 'encoded_seq.shape')
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            print(encoded_seq.shape)
            # seq expanded to batch
            encoded_seq = encoded_seq.unsqueeze(0).expand(args.batch_size, -1)
            print(frozen_embedding_model.weight.device, encoded_seq.device)
            encoded_seq_hidden = frozen_embedding_model(encoded_seq.cuda() if th.cuda.is_available() else encoded_seq)
            seqlen = encoded_seq.size(1)
            sample_shape = (args.batch_size, seqlen, args.in_channel,)

            if args.eval_task_.startswith('control'):
                print('-*' * 200, label, '-*' * 200)
                if args.use_ddim:
                    loop_func_ = diffusion.ddim_sample_loop_progressive
                else:
                    loop_func_ = diffusion.p_sample_loop_progressive

                for sample in loop_func_(
                        model,
                        sample_shape,
                        denoised_fn=partial(denoised_fn_round, args, frozen_embedding_model.cuda() if th.cuda.is_available() else frozen_embedding_model),
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs,
                        device=encoded_seq_hidden.device,
                        langevin_fn=langevin_fn_selected,
                        eta=args.eta,
                ):
                    final = sample["sample"]
                # try:
                #     import pdb
                #     pdb.set_trace()
                #     label_ids = th.tensor([label]).expand(args.batch_size).cuda()
                #     tgt_embs = frozen_embedding_model(label_ids[:, final.size(1):])
                #
                #     label_ids2 = th.cat([label_ids[:, :final.size(1)], label_ids], dim=1)
                #     label_ids2[:, :64 * 2 + 1] = -100
                #     tt = th.LongTensor([0]).expand(final.size(0)).to(final.device)
                #     prev_sample = diffusion.q_sample(final, tt)
                #     input_embs = th.cat([final, prev_sample, tgt_embs], dim=1)
                #     model_out = model_control(imput_embed=input_embs, labels=label_ids2)
                #     print(model_out.loss, 'final end')
                #     loss_fn = th.nn.CrossEntropyLoss(reduction='none')
                #     shifted_logits = model_out.logits[:, :-1].contiguous()
                #     shifted_labels = label_ids2[:, 1:].contiguous()
                #     loss = loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)),
                #                    shifted_labels.view(-1)).reshape(shifted_labels.shape)
                #     print(loss.sum(dim=-1).tolist())
                #     word_lst = rounding_func(args.experiment, final, frozen_embedding_model, tokenizer)
                #     print(len(word_lst))
                #     for ww, ll in zip(word_lst, loss.sum(dim=-1).tolist()):
                #         print([ww], ll)
                # except Exception as e:
                #     import pdb
                #     pdb.set_trace()

            sample = final
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            logger.log(f"created {len(all_images) * args.batch_size} samples")

        dist.barrier()
        logger.log("sampling complete")

        print('decoding for e2e', )
        print(sample.shape)
        x_t = sample
        logits = model.get_logits(x_t)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1)
        save_results(args, sample, tokens_list_to_midi_list(args, cands.indices), config.id2label[label])

    # args.out_path2 = out_path2
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
    main_args = main()
    # import numpy as np

    # if args.verbose != 'pipe':
    #     eval(args)

