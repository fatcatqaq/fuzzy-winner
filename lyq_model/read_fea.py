import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from model_config import *



class MangoDataset(Dataset):
    def __init__(self, data_dir, dayid, max_seq_len=100):
        self.df = pd.read_parquet(f'{data_dir}/train_day{dayid}_dl.parquet')
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = {}

        # 处理类别特征
        for feat in category_features:
            sample[feat] = torch.tensor(row[feat], dtype=torch.long) + 1
            #print(f"{feat} loaded")

        # 处理序列特征
        for feat in sequence_features:
            seq = json.loads(row[feat])[:self.max_seq_len]
            seq = torch.tensor(seq, dtype=torch.long) + 1
            pad_len = self.max_seq_len - len(seq)
            sample[feat] = F.pad(seq, (0,pad_len), value=0)
            #print(f"{feat} loaded")

        # 处理字典特征
        for key_feat, weight_feat in dict_features.items():
            key = list(map(int, json.loads(row[key_feat])[-2*self.max_seq_len:]))
            weight = json.loads(row[weight_feat])[-2*self.max_seq_len:]
            key = torch.tensor(key, dtype=torch.long) + 1
            weight = torch.tensor(weight, dtype=torch.float32)
            pad_len = 2 * self.max_seq_len - len(key)
            sample[key_feat] = F.pad(key, (0,pad_len), value=0)
            sample[weight_feat] = F.pad(weight, (0,pad_len), value=0)
            #print(f"{key_feat} loaded")

        for label in labels:
            sample[label] = torch.tensor(row[label], dtype=torch.float32)
            #print(print(f"{label} loaded"))

        return sample

