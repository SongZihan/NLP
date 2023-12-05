import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from ParameterStorage import ParameterStorage


# 自定义数据集函数
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = torch.tensor(enc_inputs,dtype=torch.int32)
        self.dec_inputs = torch.tensor(dec_inputs,dtype=torch.int32)
        self.dec_outputs = torch.tensor(dec_outputs,dtype=torch.int32)

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.01):
        """
        patience: 经过多少个epoch没有改善后停止训练。
        min_delta: 认为改善了的最小变化。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def make_train_state():
    return {
            'learning_rate': ParameterStorage.learning_rate,
            'epochs': ParameterStorage.epochs,
            'train_loss': [],
            'valid_loss': [],
            'model_filename': ParameterStorage.model_state_file
    }

