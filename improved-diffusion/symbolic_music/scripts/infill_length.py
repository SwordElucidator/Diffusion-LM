"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import os

import numpy as np
import torch as th

from symbolic_music.rounding import tokens_list_to_midi_list
from symbolic_music.scripts.infill_util import create_embedding, create_model, prepare_args, save_results
from symbolic_music.utils import get_tokenizer
import torch.distributed as dist
from improved_diffusion.test_util import denoised_fn_round
from functools import partial
from improved_diffusion import logger


class InfillTask(object):
    def __new__(cls, args, tokenizer):
        klass = InfillTask
        if args.eval_task_ == 'length':
            klass = LengthTask
        return super(InfillTask, klass).__new__(klass)

    def prepare_partial_seq(self):
        # partial seq: some part of the input
        logger.log('load the partial sequences')
        file = './diffusion_models/diff_midi_midi_files_REMI_bar_block_rand32_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_midi/padded_tokens_list_valid.npz'
        arr = np.load(file)['arr_0']
        encoded_partial_seq = [th.LongTensor(arr[531][0: 21])]
        return encoded_partial_seq

    def __init__(self, args, tokenizer):
        self.args = args
        self.todo_pad_token = -1
        self.tokenizer = tokenizer


class LengthTask(InfillTask):
    def prepare_partial_seq(self):
        encoded_partial_seq = super(LengthTask, self).prepare_partial_seq()
        # 16 * 16 - 10
        right_length = self.args.image_size ** 2 - len(encoded_partial_seq[0])
        # fill the left part
        right_pad = th.empty(right_length).fill_(self.todo_pad_token).long()
        # seq + right padding
        encoded_partial_seq = [th.cat([seq, right_pad], dim=0) for seq in encoded_partial_seq]
        # put the tgt_len token-> end
        # encoded_partial_seq[0][args.tgt_len - 1] = tokens2id['END']
        # put tgt_len -> start
        encoded_partial_seq[0] = th.cat([
            th.tensor(encoded_partial_seq[0][:self.args.tgt_len]),
            th.tensor([self.tokenizer.vocab['PAD_None']] * (self.args.image_size ** 2 - self.args.tgt_len))
        ])
        return encoded_partial_seq


def main():
    args = prepare_args()
    model, diffusion = create_model(args)
    frozen_embedding_model = create_embedding(args, model)
    tokenizer = get_tokenizer(args)
    task = InfillTask(args, tokenizer)
    encoded_partial_seq = task.prepare_partial_seq()
    print(encoded_partial_seq[0], len(encoded_partial_seq[0]))

    logger.log("sampling...")
    for encoded_seq in encoded_partial_seq:
        all_images = []
        print(args.num_samples, encoded_seq.shape, 'encoded_seq.shape')
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            print(encoded_seq.shape)
            # seq expanded to batch
            encoded_seq = encoded_seq.unsqueeze(0).expand(args.batch_size, -1)
            print(frozen_embedding_model.weight.device, encoded_seq.device)
            # mask fill
            partial_mask_temp = (encoded_seq == task.todo_pad_token).view(args.batch_size, -1)
            encoded_seq.masked_fill_(encoded_seq == task.todo_pad_token, 3)
            # encode
            encoded_seq_hidden = frozen_embedding_model(encoded_seq.cuda() if th.cuda.is_available() else encoded_seq)
            seqlen = encoded_seq.size(1)
            partial_mask = partial_mask_temp.unsqueeze(-1).expand(-1, -1, args.in_channel)
            sample_shape = (args.batch_size, seqlen, args.in_channel,)
            loop_func_ = diffusion.p_sample_loop_progressive_infill

            # sample looping
            for sample in loop_func_(
                    model,
                    sample_shape,
                    encoded_seq_hidden,
                    partial_mask,
                    denoised_fn=partial(denoised_fn_round, args, frozen_embedding_model.cuda() if th.cuda.is_available() else frozen_embedding_model),
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

    dist.barrier()
    logger.log("sampling complete")

    print('decoding for e2e', )
    print(sample.shape)
    x_t = sample
    reshaped_x_t = x_t
    logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
    cands = th.topk(logits, k=1, dim=-1)
    save_results(args, sample, tokens_list_to_midi_list(args, cands.indices))

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
