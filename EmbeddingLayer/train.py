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
from torch.nn.utils.rnn import pad_sequence
import time
from rich import print

from processing import *
from ReviewClassifierEmbedding import *
from rich.console import Console
from rich.progress import track

#################### 参数合集 ####################
data_path = r'..\Data\reviews_with_splits_lite.csv'
model_state_file = 'model.pth'
save_dir = 'model_storage/yelp/'
# Training hyper parameters
batch_size = 128
early_stopping_criteria = 5
learning_rate = 0.003
num_epochs = 80
seed = 1337
# Runtime options
cuda = False
device = 'cpu'

if __name__ == '__main__':
    console = Console()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("[italic red]GPU Ready![/italic red]")
    else:
        device = torch.device("cpu")
        print("[italic blue]Using CPU[/italic blue]")

    # Set seed for reproducibility
    set_seed_everywhere(seed, cuda)
    # handle dirs
    handle_dirs(save_dir)

    #################### 读取数据 ####################
    with console.status("[bold green]Processing...[/bold green]") as status:
        data = pd.read_csv(data_path)
        data['label'] = data['rating'].map(lambda x: 1 if x == 'positive' else 0)
        spacy_en = spacy.load('en_core_web_sm')  # spacy 分词器

        data['processed_text'] = data['review'].apply(lambda x: tokenize_en(x,spacy_en))

        text_map,processed_text = generate_vocab(list(data['processed_text']))

        max_length = max(len(seq) for seq in processed_text)
        padded_sequences = np.array([np.pad(seq, (0, max_length - len(seq)), mode='constant') for seq in processed_text])
        data['processed_text'] = padded_sequences.tolist()
        console.log("[italic green]Data Loaded![/italic green]")
        #################### 制作数据集 ####################
        print("[bold green]Preparing data iterator[/bold green]")
        # status.update("[bold green]Preparing data iterator[/bold green]")
        # 划分数据集
        train_data = data[data['split'] == 'train']
        valid_data = data[data['split'] == 'val']
        test_data = data[data['split'] == 'test']
        print(11)
        # 构建 pytorch Dataset
        train_dataset = CustomDataset(list(train_data['processed_text']), list(train_data['label']))
        valid_dataset = CustomDataset(list(valid_data['processed_text']), list(valid_data['label']))
        test_dataset = CustomDataset(list(test_data['processed_text']), list(test_data['label']))

        console.log("[italic green]Dataset Building Complete![/italic green]")
        #################### 准备模型 ####################
        status.update("[bold green]Preparing model...[/bold green]")

        classifier = ReviewClassifierRNN(len(text_map) + 1,256,256,1,device).to(device)

        loss_func = nn.BCELoss()
        optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                         mode='min', factor=0.5,
                                                         patience=1)
        train_state = make_train_state(learning_rate,model_state_file)
        console.log("[italic green]Model Building Complete![/italic green]")

    print("[italic green]Starting training...[/italic green]")
    #################### 开始训练 ####################
    # console.log("[italic green]Starting training...[/italic green]")
    # 训练进度条
    progress = Progress()
    with Progress() as progress:
        # 创建三个不同的进度条
        total_epoch_progress = progress.add_task("[red]Total Percentage...", total=num_epochs)
        train_epoch_progress = progress.add_task("[green]Train Epoch...",
                                                 total=len(train_dataset) // batch_size)
        valid_epoch_progress = progress.add_task("[yellow]Valid Epoch...",
                                                 total=len(valid_dataset) // batch_size)

        # 开始训练
        for epoch_index in range(num_epochs):
            train_state['epoch_index'] = epoch_index

            # 构建迭代器
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                          drop_last=True)
            valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True,
                                          drop_last=True)
            # test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,
            #                              drop_last=True)
            #################### 训练模式 ####################
            running_loss = 0.0
            running_acc = 0.0
            classifier.train()  # 开启训练模式
            # 训练阶段
            for batch_index, batch_data in enumerate(train_dataloader):
                # --------------------------------------
                # step 1. zero the gradients
                optimizer.zero_grad()

                # step 2. compute the output
                y_pred = classifier.forward(x_in=batch_data[0].to(device))

                # step 3. compute the loss
                loss = loss_func(y_pred.cpu(), batch_data[1].float())
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # step 4. use loss to produce gradients
                loss.backward()

                # step 5. use optimizer to take gradient step
                optimizer.step()
                # -----------------------------------------
                # compute the accuracy
                acc_t = compute_accuracy(y_pred, batch_data[1])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                progress.update(train_epoch_progress, advance=1,
                                description=f'[green]Train Epoch...loss:{round(running_loss, 2)} Acc:{round(running_acc, 2)} Epoch:{epoch_index}')

            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)

            #################### 验证模式 ####################
            running_loss = 0.
            running_acc = 0.
            classifier.eval()

            for batch_index, batch_data in enumerate(valid_dataloader):
                # compute the output
                y_pred = classifier.forward(x_in=batch_data[0].to(device))

                # step 3. compute the loss
                loss = loss_func(y_pred.cpu(), batch_data[1].float())
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # compute the accuracy
                acc_t = compute_accuracy(y_pred, batch_data[1])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                progress.update(valid_epoch_progress, advance=1,
                                description=f'[yellow]Valid Epoch...loss:{round(running_loss, 2)} Acc:{round(running_acc, 2)} Epoch:{epoch_index}')

            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)

            train_state = update_train_state(early_stopping_criteria, model=classifier,
                                             train_state=train_state)

            scheduler.step(train_state['val_loss'][-1])

            # 重置进度条
            progress.reset(train_epoch_progress, total=batch_size)  # 重置已有任务
            progress.reset(valid_epoch_progress, total=batch_size)  # 重置已有任务

            progress.update(total_epoch_progress, advance=1)

            if train_state['stop_early']:
                break

        print("[italic red]Training complete![/italic red]")
        #################### 存储训练数据 ####################
        with open(save_dir + "/train_state.json", 'w') as f:
            json.dump(train_state, f)















