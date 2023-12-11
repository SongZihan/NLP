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

def data_gen(V, batch, nbatches):
    """
    生成随机数据用于源-目标复制任务。

    参数:
    V: 词汇表的大小（即可以生成的最大整数）
    batch: 每个批次中的样本数量
    nbatches: 要生成的批次总数

    生成的每个批次包含随机整数数组，每个数组长度固定为10。
    """

    for i in range(nbatches):  # 遍历每个批次
        # 生成随机整数数组。每个数组的大小为 [batch, 10]。
        # 数值在 1 和 V (不含V) 之间。
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))

        # 将每个数组的第一个元素设置为 1，这可能是为了表示序列的开始。
        data[:, 0] = 1

        # 创建源数据和目标数据的变量。
        # 这里源数据和目标数据是相同的，因为任务是复制。
        # requires_grad 设置为 False，表示在这些变量上不需要计算梯度，
        # 这是因为它们是输入数据，不是模型参数。
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)

        # 使用 Batch 类创建一个批次，并产出（yield）。
        # 这里的 0 可能是用于指定某种掩码或者特定的批处理配置。
        yield Batch(src, tgt, 0)


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
    # Train the simple copy task.
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch(data_gen(V, 30, 20), model,
                  SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V, 30, 5), model,
                        SimpleLossCompute(model.generator, criterion, None)))
