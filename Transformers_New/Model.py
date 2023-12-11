import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    构建序列到序列的模型。

    参数:
    src_vocab: 源语言词汇大小
    tgt_vocab: 目标语言词汇大小
    N: 编码器和解码器堆叠的层数
    d_model: 模型中的嵌入维度
    d_ff: 前馈网络中的内层维度
    h: 多头注意力机制中的头数
    dropout: dropout的比率
    """
    c = copy.deepcopy  # 用于创建深度拷贝，避免共享相同的实例
    # 创建多头注意力层
    attn = MultiHeadedAttention(h, d_model)
    # 创建前馈网络
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 创建位置编码
    position = PositionalEncoding(d_model, dropout)
    # 构建模型
    model = EncoderDecoder(
        # 编码器部分
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        # 解码器部分
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        # 源语言嵌入层，加上位置编码
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        # 目标语言嵌入层，加上位置编码
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        # 生成器部分
        Generator(d_model, tgt_vocab))

    # 初始化模型参数
    # 使用Glorot初始化（也称为Xavier均匀初始化）
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model

class EncoderDecoder(nn.Module):
    """
    标准的编码器-解码器架构。许多其他模型的基础。
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        初始化编码器-解码器模型。
        参数:
        encoder - 编码器对象
        decoder - 解码器对象
        src_embed - 源序列嵌入函数
        tgt_embed - 目标序列嵌入函数
        generator - 生成器对象，用于从解码器输出生成最终结果
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        encode_result = self.encode(src, src_mask)
        return self.decode(encode_result, src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    定义标准的线性+Softmax生成步骤。Generator 类是模型的最后一步，它将解码器的输出转换为最终的词汇表分布。
    """

    def __init__(self, d_model, vocab):
        """
        初始化生成器。
        参数:
        d_model - 嵌入的维度
        vocab - 词汇表的大小
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        "前向传播方法，应用线性变换和log_softmax。"
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers. clones 函数用于创建特定模块的多个副本，这在构建具有多层结构的编码器和解码器时非常有用。"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        """
        层归一化（Layer Normalization）模块。

        层归一化通常用于自然语言处理和其他深度学习应用中，有助于稳定训练过程。
        它与批归一化（Batch Normalization）不同，因为层归一化是对单个样本中的所有特征进行归一化，
        而不是跨批次（batch）的特征。

        参数:
        features (int): 输入特征的维度。
        eps (float): 添加到分母中的小值，以提高数值稳定性。默认为1e-6。
        """
        super(LayerNorm, self).__init__()
        # 初始化可学习的参数 a 和 b，分别用于缩放和平移
        self.a_2 = nn.Parameter(torch.ones(features))  # 缩放参数
        self.b_2 = nn.Parameter(torch.zeros(features))  # 平移参数
        self.eps = eps  # 小值，用于数值稳定性

    def forward(self, x):
        """
        前向传播方法。

        参数:
        x (Tensor): 输入特征张量，形状为 [..., features]。

        返回:
        Tensor: 归一化后的特征张量。
        """
        # 计算输入的均值和标准差
        mean = x.mean(-1, keepdim=True)  # 最后一个维度的均值
        std = x.std(-1, keepdim=True)  # 最后一个维度的标准差

        # 应用层归一化：缩放和平移
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Encoder(nn.Module):
    """
    Transformer模型的核心编码器部分，由N层堆叠构成。
    参数:
    layer (nn.Module): 编码器层的实例，这个层将被复制N次来构建编码器。
    N (int): 编码器中层的数量。
    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 克隆给定的层N次
        self.layers = clones(layer, N)
        # 层归一化
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        前向传播方法。

        参数:
        x (Tensor): 输入张量。
        mask (Tensor): 应用于输入的掩码张量。

        返回:
        Tensor: 编码器输出。
        """
        # 依次通过每一层
        for layer in self.layers:
            x = layer(x, mask)
        # 应用最后的层归一化
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    子层连接，包括残差连接和层归一化。
    参数:
    size (int): 输入的特征维数。
    dropout (float): dropout概率。
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)  # 层归一化
        self.dropout = nn.Dropout(dropout)  # Dropout层
    def forward(self, x, sublayer):
        """
        前向传播方法。
        参数:
        x (Tensor): 输入张量。
        sublayer (function): 要应用的子层函数。
        返回:
        Tensor: 经过子层处理和残差连接后的输出。
        """
        # 应用子层处理和残差连接
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    Transformer编码器的单个层，包含自注意力机制和前馈网络。
    参数:
    size (int): 输入特征维度。
    self_attn (nn.Module): 自注意力模块。
    feed_forward (nn.Module): 前馈网络模块。
    dropout (float): dropout概率。
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn  # 自注意力模块
        self.feed_forward = feed_forward  # 前馈网络
        # 创建两个子层连接
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        前向传播方法。
        参数:
        x (Tensor): 输入张量。
        mask (Tensor): 应用于自注意力的掩码张量。
        返回:
        Tensor: 编码器层的输出。
        """
        # 应用自注意力和前馈网络
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """
    Transformer模型的解码器部分，由N层堆叠构成，并支持掩码操作。
    参数:
    layer (nn.Module): 解码器层的实例，这个层将被复制N次来构建解码器。
    N (int): 解码器中层的数量。
    """

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # 克隆给定的层N次
        self.layers = clones(layer, N)
        # 层归一化
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        前向传播方法。
        参数:
        x (Tensor): 输入张量，来自于前一个解码器层或嵌入层。
        memory (Tensor): 编码器的输出。
        src_mask (Tensor): 应用于源序列的掩码张量。
        tgt_mask (Tensor): 应用于目标序列的掩码张量。
        返回:
        Tensor: 解码器输出。
        """
        # 依次通过每一层
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        # 应用最后的层归一化
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Transformer解码器的单个层，包含自注意力机制、源注意力机制和前馈网络。
    参数:
    size (int): 输入特征维度。
    self_attn (nn.Module): 自注意力模块。
    src_attn (nn.Module): 源注意力模块，关注编码器的输出。
    feed_forward (nn.Module): 前馈网络模块。
    dropout (float): dropout概率。
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn  # 自注意力模块
        self.src_attn = src_attn  # 源注意力模块
        self.feed_forward = feed_forward  # 前馈网络
        # 创建三个子层连接
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        前向传播方法。
        参数:
        x (Tensor): 输入张量，来自于前一个解码器层或嵌入层。
        memory (Tensor): 编码器的输出。
        src_mask (Tensor): 应用于源序列的掩码张量。
        tgt_mask (Tensor): 应用于目标序列的掩码张量。

        返回:
        Tensor: 解码器层的输出。
        """
        m = memory
        # 通过自注意力层
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 通过源注意力层，关注编码器输出
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 通过前馈网络
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    """
    生成用于掩蔽后续位置的掩码。
    这个掩码用于Transformer模型的解码器，确保在预测序列的每个位置时，
    只能使用该位置之前的位置的信息，而不能使用后面的信息。
    参数:
    size (int): 掩码的大小，通常是目标序列的长度。
    返回:
    Tensor: 一个三维的掩码张量，形状为 (1, size, size)。
    """
    # 定义掩码的形状。这里创建了一个三维张量，
    # 第一个维度是批次大小（在这里固定为1），其他两个维度都是掩码大小。
    attn_shape = (1, size, size)

    # 使用numpy.triu（上三角矩阵）函数创建掩码。
    # k=1意味着对角线以上的元素（不包括对角线）被设置为1，其他元素为0。
    # 这样生成的矩阵用于屏蔽每个位置之后的位置。
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 将numpy数组转换为PyTorch张量，并将1转换为0，0转换为1。
    # 在Transformer模型中，通常使用0表示不屏蔽，1表示屏蔽。
    k = torch.from_numpy(subsequent_mask) == 0
    return k


def attention(query, key, value, mask=None, dropout=None):
    """
    计算'缩放点积注意力'。
    缩放点积注意力是Transformer模型的核心组成部分之一，它通过计算查询（query）和键（key）之间的点积来获取注意力权重，
    然后应用这些权重于值（value）。
    参数:
    query (Tensor): 查询张量。
    key (Tensor): 键张量。
    value (Tensor): 值张量。
    mask (Tensor, 可选): 掩码张量，用于屏蔽无关的位置。
    dropout (nn.Module, 可选): Dropout层，用于正则化。
    返回:
    tuple: 包含两个元素的元组。
          - 第一个元素是注意力机制的输出。
          - 第二个元素是注意力权重。
    """
    # 获取查询的最后一个维度的大小，用于缩放点积
    d_k = query.size(-1)

    # 计算查询和键的点积，然后缩放
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 如果提供了掩码，则将掩码位置的分数设置为非常小的负数（在softmax之前），这样这些位置的权重接近于零
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 对分数应用softmax，获取最终的注意力权重
    p_attn = F.softmax(scores, dim=-1)

    # 如果提供了dropout，则应用dropout进行正则化
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 使用注意力权重加权值张量
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    实现多头注意力机制的模块。

    在多头注意力机制中，注意力机制被分割成多个“头”，每个头独立学习输入的不同表示，
    最后将这些表示组合起来形成最终的输出。

    参数:
    h (int): 多头注意力机制中的头数。
    d_model (int): 输入特征的维度。
    dropout (float, 可选): Dropout概率。
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 确保模型的维度可以被头数整除
        assert d_model % h == 0

        # 我们假设d_v（值的维度）总是等于d_k（键的维度）
        self.d_k = d_model // h
        self.h = h
        # 创建四个线性层，用于后续的线性变换
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # 用于存储注意力权重
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        前向传播方法。
        参数:
        query (Tensor): 查询张量。
        key (Tensor): 键张量。
        value (Tensor): 值张量。
        mask (Tensor, 可选): 掩码张量。
        返回:
        Tensor: 多头注意力的输出结果。
        """
        if mask is not None:
            # 同一个掩码应用于所有的头
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) 对查询、键、值进行线性变换，并重新调整形状以便于多头注意力
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        # 2) 在所有投影向量上应用注意力机制
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) 将多头注意力的结果拼接起来，然后应用最后一个线性变换
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    """
    Transformer模型输入的嵌入层类。
    属性:
        lut (nn.Embedding): 嵌入层，将输入的标记映射为嵌入向量。
        d_model (int): 嵌入向量的维度。
    参数:
        d_model (int): 嵌入向量的大小。
        vocab (int): 词汇表的大小（不同标记的数量）。
    """

    def __init__(self, d_model, vocab):
        """
        初始化Embeddings类。
        参数:
            d_model (int): 嵌入向量的维度。
            vocab (int): 词汇表的大小。
        """
        super(Embeddings, self).__init__()  # 调用nn.Module的初始化方法
        self.lut = nn.Embedding(vocab, d_model)  # 创建嵌入层
        self.d_model = d_model  # 存储嵌入向量维度

    def forward(self, x):
        """
        定义模型的前向传播逻辑。
        参数:
            x (Tensor): 输入到嵌入层的整数索引（通常是一批次的词汇索引）。
        返回:
            Tensor: 缩放后的嵌入向量。
        """
        return self.lut(x) * math.sqrt(self.d_model)  # 将输入映射为嵌入向量并进行缩放


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        # 创建顺序列表
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)





class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


if __name__ == '__main__':
    a = subsequent_mask(5)
