# 该文件用来处理训练数据
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
import time

from processFunction import *
from rich import print
from rich.console import Console
from rich.progress import track

args = Namespace(
    data_path=r'..\Data\reviews_with_splits_lite.csv',
    vector_storage_path=r".\vector_storage",
    vector_name = 'vectors.csv',
    # chunk_size = 20000,
    max_sentence_length=200,
    device=None,
)

if __name__ == '__main__':
    console = Console()
    with console.status("[bold green]Working on processing...") as status:
        handle_dirs(args.vector_storage_path)

        data = pd.read_csv(args.data_path)
        vector_dict = load_vectors(r"..\Data\wiki-news-300d-1M.vec")
        unkown_word = np.random.randn(len(vector_dict['random']))  # 生成一个随机向量表示为所有不在词典中的词
        mask_word = np.random.randn(len(vector_dict['random']))  # 生成一个随机向量表示为所有不在词典中的词
        spacy_en = spacy.load('en_core_web_sm')  # spacy 分词器

    print("[bold red]Start Process![/bold red]")
    review_processed = []
    for x in track(data['review'], "[bold green]UnderProcessing..[/bold green]"):
        review_processed.append(encoding_vectors(tokenize_en(x, spacy_en), vector_dict, unkown_word,mask_word,args.max_sentence_length))
    new_data = pd.DataFrame()
    new_data['label'] = data['rating'].map(lambda x: 1 if x == 'positive' else 0)
    new_data['vector'] = review_processed
    new_data['split'] = data['split']
    new_data.to_csv(args.vector_storage_path + f'/{args.vector_name}')

    print("[bold yellow]Complete![/bold yellow]")

