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


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = tokens[1:]
    return data


def encoding_vectors(token, vector_dict, unkown_word, mask_word, max_length):
    """
    将分好的单词转化成为embedding vector
    """
    embedding_vector = []

    token = token[:max_length]

    for i in range(max_length):
        try:
            embedding_vector.append(vector_dict[token[i]])
        except Exception as e:
            if e == KeyError:
                embedding_vector.append(unkown_word)
    # 填充mask
    if len(embedding_vector) < max_length:
        diff_len = max_length - len(embedding_vector)
        for k in range(diff_len):
            embedding_vector.append(mask_word)

    # 使用numpy将字符串转化为浮点数，再转化为tensor后展平
    return torch.from_numpy(np.asarray(embedding_vector, dtype=float))


def read_data_in_chunks(file_path, chunk_size=10000):
    """
    使用 Pandas 逐块读取数据。
    :param file_path: CSV文件路径。
    :param chunk_size: 每次读取的行数。
    :return: DataFrame的迭代器。
    """
    return pd.read_csv(file_path, chunksize=chunk_size)
def write_data_chunk(output_file_path, data_chunk, mode='a'):
    """
    将处理后的数据块写入CSV文件。
    :param output_file_path: 输出文件的路径。
    :param data_chunk: DataFrame数据块。
    :param mode: 文件打开模式，默认为追加('a')。
    """
    header = mode == 'w'  # 如果是写入模式，则添加表头
    data_chunk.to_csv(output_file_path, mode=mode, header=header, index=False)

# 创建自定义的Dataset对象
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        """
        :param data: 特征数据
        :param labels: 标签
        :param vector_dict: 用于embedding的字典
        :param unkown_word: 非字典字符应用unkown word
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        return self.data[idx], label




# dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
#                         shuffle=shuffle, drop_last=drop_last)


# 训练参数管理
def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'model_filename': args.model_state_file}


def update_train_state(args, model, train_state):
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
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state


def compute_accuracy(y_pred, y_target):
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred) > 0.5).cpu().long()  # .max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    # 这一行计算准确率。它将正确预测的数量除以总预测数（len(y_pred_indices)），然后乘以 100，将其转换为百分比形式。
    return n_correct / len(y_pred_indices) * 100
