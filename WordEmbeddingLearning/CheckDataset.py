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
from rich import print

from ReviewClassifier import ReviewClassifier
from processFunction import *
from rich.console import Console
from rich.progress import track



if __name__ == '__main__':
    data = pd.read_csv(r"W:\PythonDoc\NLP\Data\reviews_with_splits_lite.csv")
    spacy_en = spacy.load('en_core_web_sm')  # spacy 分词器
    data['word_tokenized'] = data['review'].map(lambda x: tokenize_en(x, spacy_en))


    len_list = [len(x) for x in data['word_tokenized']]

    print(f"mean: {np.mean(len_list)}, max: {max(len_list)}, min: {min(len_list)}")




