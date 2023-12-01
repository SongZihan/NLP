import torch
import math
import matplotlib.pyplot as plt

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 创建一个足够长的位置编码矩阵
        pe = torch.zeros(max_len, d_model)

        # 计算位置编码 对应论文中3.5节
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # 生成一个从0到d_model的列表，并且增加一个维度，形如：[[0],[1],,,[d_model]]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        # 注册为常量，不参与训练
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码添加到输入的词嵌入中
        x = x + self.pe[:x.size(0), :]
        return x

# 创建模型实例
d_model = 512 # 嵌入维度
max_len = 100 # 最大句子长度
pe = PositionalEncoding(d_model, max_len)

# 创建一个假设的词嵌入序列
batch_size = 5
seq_len = 60
embedding = torch.randn(batch_size, seq_len, d_model)

# 应用位置编码
encoded = pe(embedding)

# 查看位置编码
plt.figure(figsize=(15, 5))
plt.imshow(pe.pe.squeeze().numpy())
plt.title("Positional Encoding")
plt.xlabel("Embedding Dimensions")
plt.ylabel("Sentence Position")
plt.colorbar()
plt.show()