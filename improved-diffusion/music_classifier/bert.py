import argparse
import os
from collections import Counter

import numpy as np
import torch
from datasets import Dataset
from miditoolkit import MidiFile

from improved_diffusion.script_util import add_dict_to_argparser
from music_classifier.easy_bert_classifier import BertNetForSequenceClassification, BertNetForPreTraining
from music_classifier.transfomer_net import TransformerNetClassifierModel
from symbolic_music.advanced_padding import advanced_remi_bar_block
from symbolic_music.utils import get_tokenizer
from transformers import BertConfig, BertModel, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollator, DataCollatorWithPadding, IntervalStrategy


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


def create_model(data_args, num_labels, id2label, label2id):
    config = BertConfig.from_pretrained("bert-base-uncased")
    config.num_labels = num_labels
    tokenizer = get_tokenizer(data_args)
    config.vocab_size = len(tokenizer.vocab)
    config.label2id = label2id
    config.id2label = id2label
    config.to_json_file(os.path.join(data_args.output_path, 'bert-config.json'))
    model = TransformerNetClassifierModel(config, data_args.input_emb_dim, 128)
    if torch.cuda.is_available():
        weight = torch.load(data_args.path_learned)
    else:
        weight = torch.load(data_args.path_learned, map_location=torch.device('cpu'))
    model.transformer_net.load_state_dict(weight)
    model.transformer_net.word_embedding.weight.requires_grad = False
    return model


def train(data_args, data_train, data_valid, num_labels, id2label, label2id):
    model = create_model(data_args, num_labels, id2label, label2id)
    training_args = TrainingArguments(
        output_dir=data_args.output_path,
        # learning_rate=1e-4,
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
    )
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
        batch_size=64,
        task='train',
        path_learned='./diffusion_models/diff_midi_midi_files_REMI_bar_block_rand32_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_midi/model200000.pt'
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
    data_train = [{"label": y, "input_ids": torch.tensor(x), "timesteps": 0} for x, y in zip(x_train, y_train_cleaned)]
    data_valid = [{"label": y, "input_ids": torch.tensor(x), "timesteps": 0} for x, y in zip(x_valid, y_valid_cleaned)]
    return data_train, data_valid, len(large_indexes) + 1, id2label, label2id


if __name__ == '__main__':
    args = create_argparser().parse_args()
    if args.task == 'train':
        train(args, *create_data(args))
