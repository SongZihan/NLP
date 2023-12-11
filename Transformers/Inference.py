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
    with open(ParameterStorage.storage_path, 'rb') as f:
        data = pickle.load(f)
    test_loader = Data.DataLoader(
        MyDataSet(data['test']['encoder_input'], data['test']['decoder_input'], data['test']['decoder_output']),
        2, True)
    vocab = Vocabulary()
    vocab.from_series()

    vocab_size_src = len(vocab.text_dict_src)
    vocab_size_trt = len(vocab.text_dict_trt)

    model = Transformer(vocab_size_src, vocab_size_trt).to(ParameterStorage.device)
    model.load_state_dict(torch.load(ParameterStorage.model_state_file))
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 占位符 索引为0.
    model.eval()
    #################### 测试集 ####################
    result_dict = {}
    for enc_inputs, dec_inputs, dec_outputs in test_loader:  # enc_inputs : [batch_size, src_len]

        input_words = vocab.ids_to_texts(enc_inputs.detach().numpy(),IsDecoderData=False)

        # 验证时不使用deco_inputs,自回归以计算性能
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(ParameterStorage.device), dec_inputs.to(
            ParameterStorage.device), dec_outputs.to(ParameterStorage.device)
        predicts, predicts_loss = model.forward(enc_inputs, dec_inputs, True)

        output_words = vocab.ids_to_texts(predicts.cpu().detach().numpy())

        loss = criterion(predicts_loss.to(ParameterStorage.device), dec_outputs.view(-1).long())
        for k in range(len(input_words)):
            result_dict[len(result_dict)] = {"input":' '.join(input_words[k]),"output":' '.join(output_words[k]),"loss":loss.item()}
        print("+1")


    # 存储参数
    with open(ParameterStorage.test_performance_file, 'w') as json_file:
        json.dump(result_dict, json_file)

