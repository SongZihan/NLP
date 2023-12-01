import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


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