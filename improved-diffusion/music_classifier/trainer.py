import argparse
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import torch
from miditoolkit import MidiFile

from improved_diffusion.script_util import add_dict_to_argparser, create_model_and_diffusion, \
    model_and_diffusion_defaults, args_to_dict
from music_classifier.simplified_transformer_net import SimplifiedTransformerNetClassifierModel
from music_classifier.transfomer_net import TransformerNetClassifierModel, TimedTransformerNetModelForPretrain
from symbolic_music.advanced_padding import advanced_remi_bar_block
from symbolic_music.utils import get_tokenizer
from transformers import BertConfig, TrainingArguments, Trainer, IntervalStrategy, get_cosine_schedule_with_warmup, \
    AdamW, DataCollatorForLanguageModeling, AutoTokenizer, BatchEncoding
from transformers.data.data_collator import DataCollatorMixin, InputDataClass


def create_dataset(data_args, split='train'):
    data_path = os.path.join(data_args.output_path, f'{split}_data.npz')
    if os.path.exists(data_path):
        data = np.load(data_path)
        return data['arr_0'], data['arr_1']
    tokenizer = get_tokenizer(data_args)
    x, y = [], []
    for midi_file_name in os.listdir(os.path.join(data_args.data_path, split)):
        if midi_file_name.endswith('.mid'):
            midifile = MidiFile(os.path.join(data_args.data_path, split, midi_file_name))
            tokens = tokenizer.midi_to_tokens(midifile)
            ins = midifile.instruments[0].program
            if data_args.padding_mode == 'bar_block':
                for block in advanced_remi_bar_block(tokens, data_args.image_size ** 2, skip_paddings_ratio=0.2):
                    x.append(block)
                    y.append(ins)
            else:
                raise NotImplementedError
    np.savez(data_path, x, y)
    return x, y


ALLOWED_TYPES = {
    'polka',
    'sonatina',
    'etude',
    'rondo',
    'ballade',
    'fantasia',
    'christian',
    'variations',
    'prelude',
    'valse',
    'waltz',
    'morceaux',
    'mazurka',
    'sonata',
    'romance',
}


def create_giant_dataset(data_args, split):
    data_path = os.path.join(data_args.output_path, f'{split}_giant_data.npz')
    if os.path.exists(data_path):
        data = np.load(data_path)
        return data['arr_0'], data['arr_1']
    tokenizer = get_tokenizer(data_args)
    x, y = [], []
    for midi_file_name in os.listdir(os.path.join(data_args.data_path, split)):
        if midi_file_name.endswith('.mid'):
            lowered = midi_file_name.lower()
            type_ = 'unknown'
            for t in ALLOWED_TYPES:
                if t in lowered:
                    type_ = t
                    break
            midifile = MidiFile(os.path.join(data_args.data_path, split, midi_file_name))
            tokens = tokenizer.midi_to_tokens(midifile)
            if data_args.padding_mode == 'bar_block':
                for block in advanced_remi_bar_block(tokens, data_args.image_size ** 2, skip_paddings_ratio=0.2):
                    x.append(block)
                    y.append(type_)
            else:
                raise NotImplementedError
    np.savez(data_path, x, y)
    print(Counter(y))
    return x, y


def make_simpler_config(data_args, config):
    config.num_hidden_layers = 6
    config.hidden_size = data_args.input_emb_dim
    config.num_attention_heads = 8
    config.intermediate_size = config.hidden_size * 4
    config.max_position_embeddings = 1024
    config.position_embedding_type = 'relative_key'


# def make_normal_config(data_args, config):
#     config.num_hidden_layers = 6
#     config.hidden_size = 256
#     config.num_attention_heads = 8
#     config.intermediate_size = config.hidden_size * 4
#     config.max_position_embeddings = 1024
#     config.position_embedding_type = 'relative_key'


def create_model(data_args, num_labels, id2label, label2id, is_eval=False):
    config = BertConfig.from_pretrained("bert-base-uncased")
    config.num_labels = num_labels
    tokenizer = get_tokenizer(data_args)
    config.vocab_size = len(tokenizer.vocab)
    config.label2id = label2id
    config.id2label = id2label
    if data_args.model_type == 'simplified':
        make_simpler_config(data_args, config)
    config.to_json_file(os.path.join(data_args.output_path, 'bert-config.json'))
    with open(os.path.join(*os.path.split(data_args.path_learned)[:-1], 'training_args.json'), 'r') as f:
        train_config = json.load(f)
    temp_dict = model_and_diffusion_defaults()
    temp_dict.update(train_config)
    _, diffusion = create_model_and_diffusion(**temp_dict)

    if data_args.model_type == 'simplified':
        model = SimplifiedTransformerNetClassifierModel(config, diffusion)
    else:
        model = TransformerNetClassifierModel(config, data_args.input_emb_dim, diffusion)

    if data_args.pretrained_model_path:
        weight = torch.load(data_args.pretrained_model_path)
        model.load_state_dict(weight, strict=False)
    else:
        if torch.cuda.is_available():
            weight = torch.load(data_args.path_trained if is_eval else data_args.path_learned)
            learned_embeddings = torch.load(args.path_learned)['word_embedding.weight']
        else:
            weight = torch.load(data_args.path_trained if is_eval else data_args.path_learned,
                                map_location=torch.device('cpu'))
            learned_embeddings = torch.load(args.path_learned, map_location=torch.device('cpu'))['word_embedding.weight']
        if is_eval:
            model.load_state_dict(weight)
        else:
            if data_args.model_type == 'normal':
                model.transformer_net.load_state_dict(weight, strict=False)
            else:
                model.transformer_net.word_embedding.weight.data = learned_embeddings.clone()

    if data_args.from_state_path and not is_eval:
        print(f'load state from {data_args.from_state_path}')
        weight = torch.load(data_args.from_state_path)
        model.load_state_dict(weight)
    else:
        print('will train from scratch')
    model.transformer_net.word_embedding.weight.requires_grad = False

    return model


def train(data_args, data_train, data_valid, num_labels, id2label, label2id):
    model = create_model(data_args, num_labels, id2label, label2id)
    training_args = TrainingArguments(
        output_dir=data_args.output_path,
        learning_rate=data_args.learning_rate,
        per_device_train_batch_size=data_args.batch_size,
        per_device_eval_batch_size=data_args.batch_size,
        num_train_epochs=data_args.epoches,
        weight_decay=0.0,
        do_train=True,
        do_eval=True,
        logging_steps=1000,
        evaluation_strategy=IntervalStrategy('steps'),
        logging_strategy=IntervalStrategy('steps'),
        save_steps=5000,
        seed=102,
    )

    def compute_metrics(eval_prediction):
        predictions, label_ids = eval_prediction
        import pdb
        pdb.set_trace()
        acc = np.sum(np.argmax(predictions[0], axis=1) == label_ids) / len(label_ids)
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_valid,
        compute_metrics=compute_metrics,
    )
    if data_args.from_check_point:
        print('from check point...')
        trainer.train(data_args.from_check_point)
    else:
        trainer.train()



def create_argparser():
    defaults = dict(
        midi_tokenizer='REMI',
        image_size=16,
        input_emb_dim=32,
        data_path='../datasets/midi/midi_files',
        output_path='./classifier_models/bert/',
        padding_mode='bar_block',
        epoches=30,
        learning_rate=1e-4,
        batch_size=64,
        task='train',
        pretrained_model_path='',
        from_state_path='',
        from_check_point='',
        model_type='normal',
        experiment='instrument',
        path_trained='',  # ./classifier_models/bert/checkpoint-5000/pytorch_model.bin
        path_learned=''  # ./diffusion_models/diff_midi_midi_files_REMI_bar_block_rand32_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_midi/model200000.pt
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def create_data(args):
    x_train, y_train = create_dataset(args, 'train')
    x_valid, y_valid = create_dataset(args, 'valid')
    # 处理一下行为
    large_indexes = {y for y, count in Counter(y_train).items() if count >= 100}
    label2id, id2label = {'-1': 0}, {'0': '-1'}
    for id_, index in enumerate(large_indexes):
        id2label[str(id_ + 1)] = str(index)
        label2id[str(index)] = id_ + 1
    y_train_cleaned = [label2id[(str(y) if y in large_indexes else '-1')] for y in y_train]
    y_valid_cleaned = [label2id[(str(y) if y in large_indexes else '-1')] for y in y_valid]
    data_train = [{"label": y, "input_ids": torch.tensor(x)} for x, y in zip(x_train, y_train_cleaned)]
    data_valid = [{"label": y, "input_ids": torch.tensor(x)} for x, y in zip(x_valid, y_valid_cleaned)]
    return data_train, data_valid, len(large_indexes) + 1, id2label, label2id


def create_smaller_train(x, y):
    x_processed, y_processed = [], []
    counter_y = dict(Counter(y))
    used_dict = defaultdict(int)
    max_ = int(len(y) / len(counter_y) * 1.2)
    for x_, y_ in zip(x, y):
        if used_dict[y_] > max_:
            continue
        else:
            used_dict[y_] += 1
            x_processed.append(x_)
            y_processed.append(y_)
    print(Counter(y_processed))
    return x_processed, y_processed



def create_smaller_valid(x, y):
    x_processed, y_processed = [], []
    used_dict = defaultdict(int)
    max_ = 1000
    for x_, y_ in zip(x, y):
        if used_dict[y_] > max_:
            continue
        else:
            used_dict[y_] += 1
            x_processed.append(x_)
            y_processed.append(y_)
    print(Counter(y_processed))
    return x_processed, y_processed


def create_giant_data(args):
    x_train, y_train = create_smaller_train(*create_giant_dataset(args, 'train'))
    x_valid, y_valid = create_smaller_valid(*create_giant_dataset(args, 'valid'))
    print(Counter(y_valid))
    # data aug
    large_indexes = set(y_train)
    label2id, id2label = {}, {}
    for id_, index in enumerate(large_indexes):
        id2label[str(id_)] = index
        label2id[index] = id_
    y_train_cleaned = [label2id[y] for y in y_train]
    y_valid_cleaned = [label2id[y] for y in y_valid]
    data_train = [{"label": y, "input_ids": torch.tensor(x)} for x, y in zip(x_train, y_train_cleaned)]
    data_valid = [{"label": y, "input_ids": torch.tensor(x)} for x, y in zip(x_valid, y_valid_cleaned)]
    return data_train, data_valid, len(large_indexes) + 1, id2label, label2id


def eval(data_args, data_valid, num_labels, id2label, label2id):
    print('start evaluation task...')

    model = create_model(data_args, num_labels, id2label, label2id, is_eval=True)
    loss = 0
    correct = 0
    with torch.no_grad():
        for i in range(0, len(data_valid), data_args.batch_size):
            splitted_set = data_valid[i: i + data_args.batch_size]
            input_ids = torch.cat([i['input_ids'].unsqueeze(0) for i in splitted_set])
            label = torch.tensor([i['label'] for i in splitted_set])
            timesteps = torch.tensor([i['timesteps'] for i in splitted_set])
            output = model(input_ids=input_ids, labels=label, timesteps=timesteps)
            loss += output.loss * len(input_ids)
            correct += sum(torch.argmax(output.logits, dim=1) == label)
    print(f"loss: {loss / len(data_valid)}")
    print(f"acc: {correct / len(data_valid)}")


def mdm_data_collator(features: List[InputDataClass], return_tensors, word_count) -> Dict[str, Any]:
    # should be {input_ids: xxx}
    mlm_probability = 0.1
    first = features[0]
    batch = {}

    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                # print(k, v)
                batch[k] = torch.tensor([f[k] for f in features])

    labels = batch["input_ids"].clone()
    torch.argmin(labels, dim=1)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    batch["input_ids"][indices_replaced] = 121  # 征用，没留mask

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(word_count, labels.shape, dtype=torch.long)
    batch["input_ids"][indices_random] = random_words[indices_random]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    batch["labels"] = labels
    return batch


@dataclass
class MLMDataCollator(DataCollatorMixin):
    return_tensors: str = "pt"

    def __init__(self, word_count):
        self.word_count = word_count

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors
        return mdm_data_collator(features, return_tensors, self.word_count)


def create_pretrain_model(data_args):
    config = BertConfig.from_pretrained("bert-base-uncased")
    tokenizer = get_tokenizer(data_args)
    config.vocab_size = len(tokenizer.vocab)
    config.to_json_file(os.path.join(data_args.output_path, 'bert-config.json'))
    with open(os.path.join(*os.path.split(data_args.path_learned)[:-1], 'training_args.json'), 'r') as f:
        train_config = json.load(f)
    temp_dict = model_and_diffusion_defaults()
    temp_dict.update(train_config)
    _, diffusion = create_model_and_diffusion(**temp_dict)

    model = TimedTransformerNetModelForPretrain(config, data_args.input_emb_dim, diffusion)

    if torch.cuda.is_available():
        weight = torch.load(data_args.path_learned)
    else:
        weight = torch.load(data_args.path_learned,
                            map_location=torch.device('cpu'))
    model.transformer_net.load_state_dict(weight, strict=False)
    frozen_word_embedding_weight = model.transformer_net.word_embedding.weight.clone()

    if data_args.from_state_path:
        print(f'load state from {data_args.from_state_path}')
        weight = torch.load(data_args.from_state_path)
        model.load_state_dict(weight)
        model.transformer_net.word_embedding.weight = torch.nn.Parameter(frozen_word_embedding_weight)
    else:
        print('will train from scratch')
    model.transformer_net.word_embedding.weight.requires_grad = False

    return model


def pretrain(data_args):
    x_train, _ = create_giant_dataset(args, 'train')  # TODO
    x_valid, _ = create_giant_dataset(args, 'valid')  # TODO
    print(len(x_train), len(x_valid))
    data_train = [{"input_ids": torch.tensor(x)} for x in x_train]
    data_valid = [{"input_ids": torch.tensor(x)} for x in x_valid]
    random.shuffle(data_valid)
    data_valid = data_valid[0: 2048]
    model = create_pretrain_model(data_args)
    training_args = TrainingArguments(
        output_dir=data_args.output_path,
        learning_rate=data_args.learning_rate,
        per_device_train_batch_size=data_args.batch_size,
        per_device_eval_batch_size=data_args.batch_size,
        num_train_epochs=data_args.epoches,
        weight_decay=0.0,
        do_train=True,
        do_eval=True,
        logging_steps=1000,
        evaluation_strategy=IntervalStrategy('steps'),
        logging_strategy=IntervalStrategy('steps'),
        save_steps=5000,
        seed=102,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_valid,
        data_collator=MLMDataCollator(model.config.vocab_size)
    )
    if data_args.from_check_point:
        print('from check point...')
        trainer.train(data_args.from_check_point)
    else:
        trainer.train()


if __name__ == '__main__':
    args = create_argparser().parse_args()
    if args.task == 'train':
        train(args, *(create_giant_data(args) if args.experiment == 'composition_type' else create_data(args)))
    if args.task == 'eval':
        _, data_valid, num_labels, id2label, label2id = \
            create_giant_data(args) if args.experiment == 'composition_type' else create_data(args)
        eval(args, data_valid, num_labels, id2label, label2id)
    if args.task == 'pretrain':
        pretrain(args)

