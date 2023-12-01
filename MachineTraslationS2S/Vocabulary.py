import os
from argparse import Namespace
from collections import Counter
import json
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class Vocabulary(object):
    """用于处理文本和提取词汇映射的类"""

    def __init__(self, token_to_idx=None):
        """
        参数:
            token_to_idx (dict): 一个预先存在的标记到索引的映射
        """

        # 如果没有提供映射，则初始化为空字典
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        # 创建索引到标记的映射
        self._idx_to_token = {idx: token
                              for token, idx in self._token_to_idx.items()}

    def to_serializable(self):
        """ 返回可以序列化的字典 """
        return {'token_to_idx': self._token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        """ 从序列化的字典中实例化 Vocabulary """
        return cls(**contents)

    def add_token(self, token):
        """根据令牌更新映射字典。

        参数:
            token (str): 要添加到词汇表中的项
        返回:
            index (int): 与令牌相对应的整数
        """
        # 如果令牌已存在，返回其索引，否则添加到映射中
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        """将一系列令牌添加到词汇表中

        参数:
            tokens (list): 一个字符串令牌的列表
        返回:
            indices (list): 对应于令牌的索引列表
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """检索与令牌相关联的索引

        参数:
            token (str): 要查找的令牌
        返回:
            index (int): 与令牌对应的索引
        """
        return self._token_to_idx[token]

    def lookup_index(self, index):
        """返回与索引相关联的令牌

        参数:
            index (int): 要查找的索引
        返回:
            token (str): 与索引对应的令牌
        异常:
            KeyError: 如果索引不在词汇表中
        """
        if index not in self._idx_to_token:
            raise KeyError("索引 (%d) 不在词汇表中" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", begin_seq_token="<BEGIN>",
                 end_seq_token="<END>"):

        # 初始化基类
        super(SequenceVocabulary, self).__init__(token_to_idx)

        # 设置特殊令牌
        self._mask_token = mask_token
        self._unk_token = unk_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        # 为特殊令牌添加索引
        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({'unk_token': self._unk_token,
                         'mask_token': self._mask_token,
                         'begin_seq_token': self._begin_seq_token,
                         'end_seq_token': self._end_seq_token})
        return contents


