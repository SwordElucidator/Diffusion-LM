from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
import torch
import torch.nn as nn


class TimedTransformerNetModel(nn.Module):
    def __init__(
        self,
        config,
        in_channels,  # embedding size for the notes  (channels of input tensor)   e.g. 16 / 32 / 128
        out_channels,  # output channels (embedding size) = in_channels (since discrete data)
        diffusion
    ):
        super().__init__()

        self.in_channels = in_channels
        self.diffusion = diffusion  # add diffusion to the net model
        self.train_diff_steps = 2000  # TODO config

        # embedding layer  shape -> [*shape, in_channels]
        self.word_embedding = nn.Embedding(config.vocab_size, self.in_channels)
        # language model head   in_channels -> vocab_size
        self.lm_head = nn.Linear(self.in_channels, config.vocab_size)
        with torch.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        self.time_embeddings = nn.Embedding(self.train_diff_steps + 1, config.hidden_size)
        # in_channels -> 768(hidden_size) -> 768(hidden_size)
        self.input_up_proj = nn.Sequential(
            nn.Linear(in_channels, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        print(config)
        # 下述BertLayer * 12
        # 768 ->
        # attention(SelfAttention + output(dense + LayerNorm + drop)) + 放大层dense + output(dense + LayerNorm + drop)
        # -> 768
        self.input_transformers = BertEncoder(config)
        # self.position_ids
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        # position embedding = 512 -> 768
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 768 -> 768 -> 16
        self.output_down_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, out_channels)
        )

    def get_embeds(self, input_ids):
        # shape -> [*shape, in_channels]
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        # in_channels (~16) -> vocab_size
        return self.lm_head(hidden_repr)

    def get_hidden_state(self, x, timesteps=None):
        if self.diffusion is not None:
            # sample t
            t = torch.randint(-1, self.train_diff_steps, (x.shape[0],)).to(x.device)
            t_mask = (t >= 0)
            input_embs_rand = self.diffusion.q_sample(x, t)
            x[t_mask] = input_embs_rand[t_mask]
            t[~t_mask] = self.train_diff_steps
            time_emb = self.time_embeddings(t).unsqueeze(1)

        elif self.diffusion is None and timesteps is not None:
            t = torch.LongTensor([timesteps]).expand(x.size(0)).to(self.device)
            time_emb = self.time_embeddings(t).unsqueeze(1)
        else:
            raise NotImplementedError
        #  timesteps  (1,2,3,4...)  ->    sine positional embedding    ->     128 -> 512 -> 768
        # in_channels (16) -> 768(hidden_size) -> 768(hidden_size)
        emb_x = self.input_up_proj(x)
        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length]

        emb_inputs = self.position_embeddings(position_ids) + emb_x + time_emb.expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        # 768 -> 768
        return self.input_transformers(emb_inputs).last_hidden_state

    def forward(self, x, timesteps=None):
        # (,768) -> (,16)
        h = self.output_down_proj(self.get_hidden_state(x, timesteps))
        h = h.type(x.dtype)
        return h


class TransformerNetClassifierModel(nn.Module):
    def __init__(self, config, in_channels, diffusion):
        super().__init__()
        # load bert config
        # config = AutoConfig.from_pretrained(config_name)
        # config.hidden_dropout_prob = dropout
        # config.max_position_embeddings = max_position_embeddings
        self.config = config
        self.num_labels = config.num_labels
        self.transformer_net = TimedTransformerNetModel(
            config, in_channels, in_channels, diffusion
        )

        self.pooler = BertPooler(config)

        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward(self, input_ids, labels, timesteps=None, imput_embed=None):
        # 只有eval的时候传timesteps
        if imput_embed is None:
            imput_embed = self.transformer_net.word_embedding(input_ids)
        hidden_state = self.transformer_net.get_hidden_state(imput_embed, timesteps)
        pooled_output = self.pooler(hidden_state)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_state,
        )
