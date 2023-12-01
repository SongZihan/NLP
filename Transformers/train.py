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


if __name__ == '__main__':
    storage_path = "./processed_data.pkl"
    with open(storage_path, 'rb') as f:
        data = pickle.load(f)
    loader = Data.DataLoader(MyDataSet(data['encoder_input'], data['decoder_input'], data['decoder_output']), ParameterStorage.batch_size, True)

    vocab = Vocabulary()
    vocab.from_series()

    vocab_size_src = len(vocab.text_dict_src)
    vocab_size_trt = len(vocab.text_dict_trt)

    model = Transformer(vocab_size_src,vocab_size_trt).to(ParameterStorage.device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 占位符 索引为0.
    optimizer = optim.Adamax(model.parameters(), lr=1e-3)

    #################### 训练 ####################
    for epoch in range(ParameterStorage.epochs):
        for enc_inputs, dec_inputs, dec_outputs in loader:  # enc_inputs : [batch_size, src_len]
            # dec_inputs : [batch_size, tgt_len]
            # dec_outputs: [batch_size, tgt_len]

            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(ParameterStorage.device), dec_inputs.to(ParameterStorage.device), dec_outputs.to(ParameterStorage.device)
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            loss = criterion(outputs, dec_outputs.view(-1))
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()