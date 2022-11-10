from symbolic_music.music_transformer_by_rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR, \
    create_encoder_by_config
from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder
import torch
import numpy as np
import torch as th
import torch.nn as nn
from improved_diffusion.nn import (
    SiLU,
    linear,
    timestep_embedding,
)


class MusicTransformerModel(nn.Module):

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        dropout=0,
        num_classes=None,
        config=None,
        config_name='bert-base-uncased',  # just for config
        training_mode='emb',
        vocab_size=None,
        experiment_mode='lm',
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)  # this will use the config from a pretrained model
            config.hidden_dropout_prob = dropout

        self.in_channels = in_channels  # embedding dim
        self.model_channels = model_channels  #
        self.dropout = dropout
        self.num_classes = num_classes

        if training_mode in ('e2e', 'e2e-simple'):
            self.word_embedding = nn.Embedding(vocab_size, self.in_channels)
            self.lm_head = nn.Linear(self.in_channels, vocab_size)
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight

        if experiment_mode == 'conditional_gen':
            self.conditional_gen = True
            self.encoder_emb = nn.Embedding(vocab_size, config.hidden_size)
            self.encoder = create_encoder_by_config(config)
            print(config, 'conditional_gen')
            config.is_decoder = True
            config.add_cross_attention = True
        elif experiment_mode == 'lm':
            self.conditional_gen = False

        time_embed_dim = model_channels * 4
        #
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_up_proj = nn.Sequential(
            nn.Linear(in_channels, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        print(config)
        self.input_transformers = BertEncoder(config) # TODO

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, out_channels))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def forward(self, x, timesteps, y=None, src_ids=None, src_mask=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # print(f'real model inputs: {timesteps}')
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.conditional_gen:
            assert src_ids is not None
            # print(src_ids.shape, 'source_ids shape')
            src_emb = self.encoder_emb(src_ids)
            # print(src_ids.shape, src_emb.shape)
            encoder_hidden_states = self.encoder(src_emb).last_hidden_state
            encoder_attention_mask = src_mask.unsqueeze(1).unsqueeze(1)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        emb_x = self.input_up_proj(x)
        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length ]
        # print(emb_x.shape, emb.shape, self.position_embeddings)
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        if self.conditional_gen:
            # print(emb_inputs.shape, encoder_hidden_states.shape, encoder_attention_mask.shape)
            input_trans_hidden_states = self.input_transformers(emb_inputs,
                                                                encoder_hidden_states=encoder_hidden_states,
                                                                encoder_attention_mask=encoder_attention_mask,
                                                                ).last_hidden_state
        else:
            input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        h = self.output_down_proj(input_trans_hidden_states)
        h = h.type(x.dtype)
        return h

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=-1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result
