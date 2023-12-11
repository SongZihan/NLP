import pandas as pd
import numpy as np
import pickle
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from ParameterStorage import ParameterStorage
from DataProcesser_Mine import *
from rich.console import Console
from rich import print
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

from Model import  *
from DataProcesser import *
class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data * norm

if __name__ == '__main__':
    console = Console()
    with console.status("[bold green]Processing...[/bold green]") as status:
        with open(ParameterStorage.storage_path, 'rb') as f:
            data = pickle.load(f)
        train_loader = Data.DataLoader(MyDataSet(data['train']['src'], data['train']['tgt']), ParameterStorage.batch_size, True)
        valid_loader = Data.DataLoader(MyDataSet(data['valid']['src'], data['valid']['tgt']), ParameterStorage.batch_size, True)
        test_loader = Data.DataLoader(MyDataSet(data['test']['src'], data['test']['tgt']), ParameterStorage.batch_size, True)

        vocab = Vocabulary()
        vocab.from_series()

        vocab_size_src = len(vocab.text_dict_src)
        vocab_size_trt = len(vocab.text_dict_trt)

        pad_idx = vocab.text_dict_trt['<pad>']
        model = make_model(vocab_size_src, vocab_size_trt, N=ParameterStorage.n_layers)
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        model.to(ParameterStorage.device)
        criterion = LabelSmoothing(size=vocab_size_trt, padding_idx=pad_idx, smoothing=0.1)
        criterion.to(ParameterStorage.device)

        model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        for epoch in range(ParameterStorage.epochs):
            model.train()
            loss = run_epoch_mine(train_loader,model,SimpleLossCompute(model.generator, criterion, model_opt),pad_idx)
            model.eval()
            valid_loss = run_epoch_mine(valid_loader,model,SimpleLossCompute(model.generator, criterion, None),pad_idx)
            # 存储模型
            torch.save(model.state_dict(), f'./model/{epoch}_model.pt')













