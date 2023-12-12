import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
from ParameterStorage import *

from Model import  *

class Batch:
    """
    在训练期间用于保存一批数据及其掩码的对象。

    属性:
    src: 源序列数据
    src_mask: 源序列的掩码
    trg: 目标序列数据
    trg_y: 偏移后的目标序列（用于预测的目标）
    trg_mask: 目标序列的掩码
    ntokens: 目标序列中的非填充元素总数
    """

    def __init__(self, src, trg=None, pad=0):
        """
        初始化Batch对象。

        参数:
        src: 源序列的张量
        trg: 目标序列的张量（如果有的话）
        pad: 用于填充的索引值
        """
        self.src = src.to(ParameterStorage.device)
        # 创建源序列掩码，遮蔽填充部分。增加一个维度以便后续操作。
        self.src_mask = (src != pad).unsqueeze(-2).to(ParameterStorage.device)

        if trg is not None:
            # 如果目标序列存在，则对目标序列进行处理
            # trg用于计算注意力时使用，从而去除最后一个元素
            self.trg = trg[:, :-1].to(ParameterStorage.device)
            # trg_y用于预测输出，因此去除第一个元素
            self.trg_y = trg[:, 1:].to(ParameterStorage.device)
            # 为目标序列创建掩码
            self.trg_mask = self.make_std_mask(self.trg, pad).to(ParameterStorage.device)
            # 计算目标序列中的非填充元素总数
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        创建一个掩码以隐藏填充和未来的单词。

        参数:
        tgt: 目标序列
        pad: 用于填充的索引值

        返回:
        tgt_mask: 目标序列的掩码
        """
        # 创建目标序列掩码，遮蔽填充部分
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # 结合后续掩码（用于屏蔽未来的单词）
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def run_epoch_mine(data_iter,model,loss_compute,pad):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, (src, tgt) in enumerate(data_iter):
        batch = Batch(src,tgt,pad)
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            # print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
            #       (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0


    return total_loss / total_tokens

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    使用贪心算法进行解码。
    Args:
    model: 用于编码和解码的模型。
    src (Tensor): 输入序列。
    src_mask (Tensor): 输入序列的掩码。
    max_len (int): 生成序列的最大长度。
    start_symbol (int): 序列开始的符号。

    Returns:
    Tensor: 解码生成的输出序列。

    该函数首先对输入序列进行编码，然后迭代地使用模型进行解码，
    在每一步选择概率最高的词作为下一个词，直到达到最大长度。
    """

    # 使用模型对输入序列进行编码
    memory = model.encode(src, src_mask)

    # 初始化输出序列 ys 为只包含开始符号的序列
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)

    # 循环生成后续词元，直到达到最大长度
    for i in range(max_len-1):
        # 解码得到输出
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        # 应用模型的生成器得到下一个词的概率分布
        prob = model.generator(out[:, -1])

        # 选择概率最高的词作为下一个词
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]

        # 将新词追加到 ys 中
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)

    return ys





global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class LabelSmoothing(nn.Module):
    "标签平滑是一种正则化技术，用于训练深度学习模型，特别是在分类任务中。它有助于防止模型对训练数据过拟合，使模型对输入数据的小变化更为鲁棒。"

    def __init__(self, size, padding_idx, smoothing=0.0):
        """
        size（分类任务中类别的总数）
        padding_idx（用于填充的索引，通常用于忽略某些特定的类别）
        smoothing（平滑参数，用于控制标签分布的平滑程度，默认为0，表示不进行平滑）
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """
        x 是模型的预测输出（通常是对数概率），
        target 是真实标签
        """
        assert x.size(1) == self.size   # 断言，确保输入 x 的第二维大小等于类别总数
        true_dist = x.data.clone()      # 克隆 x 的数据，用于创建平滑后的真实标签分布
        true_dist.fill_(self.smoothing / (self.size - 2)) # 用平滑参数填充 true_dist，确保所有非目标类别都获得一定的小概率
        true_dist.scatter_(1, target.data.unsqueeze(1).to(torch.int64), self.confidence) # 使用目标标签来更新 true_dist 中对应的位置，使得目标类别的概率接近于置信度。
        true_dist[:, self.padding_idx] = 0 # 将填充索引处的概率设置为0，表示忽略这些类别。
        mask = torch.nonzero(target.data == self.padding_idx) # 找出所有填充索引对应的位置。
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0) # 将这些位置在 true_dist 中的值设置为0。
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
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