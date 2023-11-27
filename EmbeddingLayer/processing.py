from argparse import Namespace
from collections import Counter
import json
import os
import re
import string
import spacy
import io

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rich.progress import Progress
import time


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def preprocessing_text(text):
    """
    全小写，去除标点符号
    """
    text = text.lower().strip()
    text = re.sub(f'[{string.punctuation}\n]', '', text)
    return text


def tokenize_en(text, spacy_en):
    """
    根据英文词库进行分词
    """
    result = []
    for tok in spacy_en.tokenizer(preprocessing_text(text)):
        if tok.text.replace(' ', '') == '':
            continue
        result.append(tok.text)

    return result


def generate_vocab(texts, min_frequency=10, unk_token="<UNK>"):
    """
    生成词典，返回词典和处理后的文档
    """
    text_map = {}
    # 计算单词频率
    frequency_map = Counter()
    for i in texts:
        frequency_map.update(i)

    # 制作词典
    index = 1
    for i in list(frequency_map.keys()):
        if frequency_map[i] < min_frequency:
            continue
        text_map[i] = index
        index += 1
    text_map[unk_token] = index

    # 将单词转化为数字索引
    texts_numeric_index = []
    for sentence in texts:
        this_sentence = []
        for this_word in sentence:
            if this_word in text_map:
                this_sentence.append(text_map[this_word])
            else:
                this_sentence.append(text_map[unk_token])
        texts_numeric_index.append(this_sentence)
    return text_map, texts_numeric_index
def generate_index_basedon_vocab(texts,text_map,unk_token="<UNK>"):
    # 根据训练集上的字典，将测试集和验证集也变成数字索引
    texts_numeric_index = []
    for sentence in texts:
        this_sentence = []
        for this_word in sentence:
            if this_word in text_map:
                this_sentence.append(text_map[this_word])
            else:
                this_sentence.append(text_map[unk_token])
        texts_numeric_index.append(this_sentence)
    return texts_numeric_index

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        """
        :param data: 特征数据
        :param labels: 标签
        :param vector_dict: 用于embedding的字典
        :param unkown_word: 非字典字符应用unkown word
        """
        self.data = torch.tensor(data, dtype=torch.long)
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data= self.data[idx]

        return data, label

def make_train_state(learning_rate,model_state_file):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'model_filename': model_state_file}


def update_train_state(early_stopping_criteria, model, train_state):
    """Handle the training state updates.

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # If loss worsened
        if loss_t >= train_state['early_stopping_best_val']:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= early_stopping_criteria

    return train_state


def compute_accuracy(y_pred, y_target):
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred) > 0.5).cpu().long()  # .max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    # 这一行计算准确率。它将正确预测的数量除以总预测数（len(y_pred_indices)），然后乘以 100，将其转换为百分比形式。
    return n_correct / len(y_pred_indices) * 100