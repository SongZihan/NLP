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
from helper import *
from Model import Transformer
from DataProcesser import Vocabulary
from rich.console import Console
from rich import print
import json


if __name__ == '__main__':
    console = Console()
    with console.status("[bold green]Processing...[/bold green]") as status:
        with open(ParameterStorage.storage_path, 'rb') as f:
            data = pickle.load(f)
        train_loader = Data.DataLoader(MyDataSet(data['train']['encoder_input'], data['train']['decoder_input'], data['train']['decoder_output']), ParameterStorage.batch_size, True)
        valid_loader = Data.DataLoader(MyDataSet(data['valid']['encoder_input'], data['valid']['decoder_input'], data['valid']['decoder_output']), ParameterStorage.batch_size, True)
        test_loader = Data.DataLoader(MyDataSet(data['test']['encoder_input'], data['test']['decoder_input'], data['test']['decoder_output']), ParameterStorage.batch_size, True)

        vocab = Vocabulary()
        vocab.from_series()

        vocab_size_src = len(vocab.text_dict_src)
        vocab_size_trt = len(vocab.text_dict_trt)

        model = Transformer(vocab_size_src,vocab_size_trt).to(ParameterStorage.device)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 占位符 索引为0.
        optimizer = optim.Adamax(model.parameters(), lr=1e-3)

        train_state = make_train_state()
        console.log("[italic green]Data Loaded![/italic green]")
        #################### 训练 ####################
        for epoch in range(ParameterStorage.epochs):

            model.train()
            train_loss_list = []
            valid_loss_list = []
            for enc_inputs, dec_inputs, dec_outputs in train_loader:  # enc_inputs : [batch_size, src_len]
                # dec_inputs : [batch_size, tgt_len]
                # dec_outputs: [batch_size, tgt_len]
                enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(ParameterStorage.device), dec_inputs.to(ParameterStorage.device), dec_outputs.to(ParameterStorage.device)
                outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
                # outputs: [batch_size * tgt_len, tgt_vocab_size]
                loss = criterion(outputs, dec_outputs.view(-1).long())
                train_loss_list.append(loss.item())
                # print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_state['train_loss'].append(round(np.mean(train_loss_list),2))

            model.eval()
            #################### 验证集 ####################
            for enc_inputs, dec_inputs, dec_outputs in valid_loader:  # enc_inputs : [batch_size, src_len]
                # 验证时不使用deco_inputs,自回归以计算性能
                enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(ParameterStorage.device), dec_inputs.to(ParameterStorage.device), dec_outputs.to(ParameterStorage.device)
                predicts,predicts_loss = model.forward(enc_inputs, dec_inputs,True)
                loss = criterion(predicts_loss.to(ParameterStorage.device), dec_outputs.view(-1).long())
                valid_loss_list.append(loss.item())

            train_state['valid_loss'].append(round(np.mean(valid_loss_list),2))

            console.log(f"[italic green]Epoch: {epoch}/{ParameterStorage.epochs} train loss: {round(np.mean(train_loss_list),2)}, valid loss: {round(np.mean(valid_loss_list),2)}[/italic green]")

            # 每epoch存储
            torch.save(model.state_dict(), f'./model/{epoch}_model.pt')

        #################### 存储模型 ####################
        console.log("[italic green]Training Complete![/italic green]")
        # 存储参数
        with open(ParameterStorage.history_file, 'w') as json_file:
            json.dump(train_state, json_file)




