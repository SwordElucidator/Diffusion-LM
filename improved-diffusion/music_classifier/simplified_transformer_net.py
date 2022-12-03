from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
import torch
import torch.nn as nn


class SimplifiedTimedTransformerNetModel(nn.Module):
    def __init__(
        self,
        config,
        diffusion=None
    ):
        super().__init__()

        self.diffusion = diffusion  # add diffusion to the net model
        self.train_diff_steps = 2000  # TODO config
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.time_embeddings = nn.Embedding(self.train_diff_steps + 1, config.hidden_size)
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

    def forward(self, x, timesteps=None):
        if self.diffusion is not None:
            # sample t
            t = torch.randint(-1, self.train_diff_steps, (x.shape[0],)).to(x.device)
            t_mask = (t >= 0)
            input_embs_rand = self.diffusion.q_sample(x, t)
            x[t_mask] = input_embs_rand[t_mask]
            t[~t_mask] = self.train_diff_steps
            time_emb = self.time_embeddings(t).unsqueeze(1)
        elif self.diffusion is None and timesteps is not None:
            time_emb = self.time_embeddings(timesteps).unsqueeze(1)
        else:
            raise NotImplementedError
        emb_x = x
        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length]
        emb_inputs = self.position_embeddings(position_ids) + emb_x + time_emb.expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        # 768 -> 768
        return self.input_transformers(emb_inputs).last_hidden_state


class SimplifiedTransformerNetClassifierModel(nn.Module):
    def __init__(self, config, diffusion=None):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.transformer_net = SimplifiedTimedTransformerNetModel(
            config, diffusion=diffusion
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
        hidden_state = self.transformer_net(imput_embed, timesteps)
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
