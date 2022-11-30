import torch, wandb
from transformers import PreTrainedModel, GPT2PreTrainedModel, GPT2Config, GPT2Model, BertPreTrainedModel, BertModel, \
    GPT2LMHeadModel, BertForMaskedLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, NextSentencePredictorOutput
from transformers.models.bert.modeling_bert import BertOnlyNSPHead
from torch import nn
from transformers import Trainer, GPT2PreTrainedModel, PreTrainedModel, DataCollator, TrainingArguments, EvalPrediction, \
    TrainerCallback
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
import collections
from transformers.utils import logging


class BertClassifier(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config, diffusion=None):
        super().__init__(config)
        self.transformer = BertModel(config)
        self.transformer.embeddings.word_embeddings = nn.Embedding(config.vocab_size, config.input_emb_dim, )
        self.up_proj = nn.Sequential(nn.Linear(config.input_emb_dim, config.input_emb_dim * 4), nn.Tanh(),
                                     nn.Linear(config.input_emb_dim * 4, config.hidden_size))
        self.diffusion = diffusion
        if diffusion is not None:
            self.train_diff_steps = config.train_diff_steps
        else:
            self.train_diff_steps = 200
        self.time_embeddings = nn.Embedding(self.train_diff_steps+1, config.hidden_size)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
        self.lm_head2 = nn.Linear(config.hidden_size, config.pos_vocab_size, bias=False)

    def set_input_embeddings(self, new_embeddings):
        self.transformer.embeddings.word_embeddings = new_embeddings

    def get_input_embeddings(self):
        return self.transformer.embeddings.word_embeddings

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids=None,
            label=None,
            input_embs=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            t = 200,
            t_aware=True,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_embs is None:
            input_embs = self.transformer.embeddings.word_embeddings(input_ids)  # input_embs

        if self.diffusion is not None:
            if self.train_diff_steps > 0:
                # sample t
                t = torch.randint(-1, self.train_diff_steps, (input_embs.shape[0],)).to(input_embs.device)
                t_mask = (t >= 0)
                input_embs_rand = self.diffusion.q_sample(input_embs, t)
                input_embs[t_mask] = input_embs_rand[t_mask]
                t[~t_mask] = self.train_diff_steps
                time_emb = self.time_embeddings(t).unsqueeze(1)

        if self.diffusion is None and t is not None:
            t = torch.LongTensor([t]).expand(input_embs.size(0)).to(self.device)
            time_emb = self.time_embeddings(t).unsqueeze(1)

        input_embs = self.up_proj(input_embs)
        if t_aware:
            input_embs = torch.cat([time_emb, input_embs], dim=1)

        transformer_outputs = self.transformer(
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=input_embs,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if t_aware and past_key_values is None:
            hidden_states = transformer_outputs[0][:, 1:, ]
            # print(hidden_states)
        else:
            hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # print(hidden_states.shape, self.lm_head2, self.lm_head2.weight.shape)
        lm_logits = self.lm_head2(hidden_states)
        # print(lm_logits.shape, hidden_states.shape, labels.shape)

        loss = None
        if pos_ids is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits
            shift_labels = labels
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )