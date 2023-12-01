import pandas as pd
import numpy as np
import pickle
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

attn_shape = [512, 256, 256]
subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]
subsequence_mask = torch.from_numpy(subsequence_mask).byte()
subsequence_mask.cuda()