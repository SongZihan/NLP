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

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).to(ParameterStorage.device)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


if __name__ == '__main__':
    console = Console()
    with console.status("[bold green]Processing...[/bold green]") as status:
        with open(ParameterStorage.storage_path, 'rb') as f:
            data = pickle.load(f)
        train_loader = Data.DataLoader(MyDataSet(data['train']['src'], data['train']['tgt']), ParameterStorage.batch_size, True)

        test_loader = Data.DataLoader(MyDataSet(data['test']['src'], data['test']['tgt']), ParameterStorage.batch_size, True)

        vocab = Vocabulary()
        vocab.from_series()

        vocab_size_src = len(vocab.text_dict_src)
        vocab_size_trt = len(vocab.text_dict_trt)

        pad_idx = vocab.text_dict_trt['<pad>']
        start_idx = vocab.text_dict_trt['<sos>']
        model = make_model(vocab_size_src, vocab_size_trt, N=ParameterStorage.n_layers)
        model.to(ParameterStorage.device)
        # 加载权重
        model.load_state_dict(torch.load(ParameterStorage.model_state_file))

        criterion = LabelSmoothing(size=vocab_size_trt, padding_idx=pad_idx, smoothing=0.1)
        criterion.to(ParameterStorage.device)

        # model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
        #                     torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        model.eval()
        for sample_batch in test_loader:

            sample_ids = sample_batch[0][0]
            sample = sample_ids.unsqueeze(0).to(ParameterStorage.device) # 形状(1,max_sentence_length)
            sample_mask = Variable(torch.ones_like(sample)).to(ParameterStorage.device)

            result = greedy_decode(model, sample, sample_mask, max_len=ParameterStorage.max_sentence_length, start_symbol=start_idx)
            result = result.cpu().detach().numpy().flatten()

            sample_text = vocab.ids_to_single_text(sample_ids.numpy(),False,False)
            print(" ".join(sample_text))
            word = vocab.ids_to_single_text(result,False,True)
            print(" ".join(word))

            label_ids = sample_batch[1][0].numpy()
            label_word = vocab.ids_to_single_text(label_ids,False,True)
            print(" ".join(label_word))


            break

            # valid_loss = run_epoch_mine(valid_loader,model,SimpleLossCompute(model.generator, criterion, None),pad_idx)
