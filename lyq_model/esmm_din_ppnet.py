import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.optim import Adam
from torch.nn import BCELoss, MSELoss
#from timm.models.layers import trunc_normal_
from sklearn.metrics import roc_auc_score
from read_fea import MangoDataset
from model_config import *
from model_utils import *




class ESMM_DIN(nn.Module):
    def __init__(self):
        super(ESMM_DIN, self).__init__()

        # 1. Embedding
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(
                num_embeddings = emb_num_dict.get(feat, DEFAULT_EMB_NUM),
                embedding_dim = emb_dim_dict.get(feat, DEFAULT_EMB_DIM))
            for feat in category_features
        })
        # 2. Bottom
        self.bottom_mlp = MultiLayerDNN(
            input_dim=TOTAL_EMB_DIM,
            hidden_dims=bottom_hidden_dims,
            activation="leaky_relu",
            dropout=0.2
        )
        # 3. Tower
        self.click_tower = MultiLayerDNN(
            input_dim=bottom_hidden_dims[-1],
            hidden_dims=tower_hidden_dims,
            output_dim=1,
            activation="leaky_relu",
            dropout=0.2
        )
        self.play_tower = MultiLayerDNN(
            input_dim=bottom_hidden_dims[-1],
            hidden_dims=tower_hidden_dims,
            output_dim=1,
            activation="leaky_relu",
            dropout=0.2
        )


    def forward(self, sample):
        # 1. Embedding lookup
        emb_list = []
        # category features
        for feat in category_features:
            feat_id = sample[feat]
            embed = self.embeddings[feat](feat_id)
            emb_list.append(embed)
        # dict features
        for key_feat, weight_feat in dict_features.items():
            feat_ids, weight = sample[key_feat], sample[weight_feat]
            shared_emb_feat = dict_features_map[key_feat]
            embeds = self.embeddings[shared_emb_feat](feat_ids)  # shape: [batch_size, len, embed_dim]
            # 加权求和（weight -> [batch_size, len, 1]）
            weighted_embed = (embeds * weight.unsqueeze(-1)).sum(dim=1)  # shape: [batch_size, embed_dim]
            emb_list.append(weighted_embed)
        # seq features (DIN)
        for seq_feat, target_feat in sequence_features_map.items():
            history_ids = sample[seq_feat]          # shape: [batch_size, seq_len]
            target_id = sample[target_feat]         # shape: [batch_size]
            padding_mask = (history_ids == 0)       # mask序列中填充的0
            # 读emb
            target_emb = self.embeddings[target_feat](target_id)    # shape: [batch_size, embed_dim]
            history_embs = self.embeddings[target_feat](history_ids)   # shape: [batch_size, seq_len, embed_dim]
            # 计算 attention
            weights = torch.bmm(history_embs, target_emb.unsqueeze(2))      # shape: [batch_size, seq_len, 1]
            weights = weights.masked_fill(padding_mask.unsqueeze(-1), -1e9)    # mask对应位置的权重
            weights = F.softmax(weights, dim=1)
            # 加权求和
            din_tensor = torch.bmm(history_embs.transpose(1, 2), weights).squeeze(-1)  # shape: [batch_size, embed_dim]
            emb_list.append(din_tensor)
        # concat
        x = torch.cat(emb_list, dim=1)
        # 2. Shared bottom
        x = self.bottom_mlp(x)
        # 3. Towers
        click_logit = self.click_tower(x).squeeze(-1)
        play_logit = self.play_tower(x).squeeze(-1)
        return click_logit, play_logit



class ESMM_DIN_PPNET(nn.Module):
    def __init__(self):
        super(ESMM_DIN_PPNET, self).__init__()

        # 1. Embedding
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(
                num_embeddings = emb_num_dict.get(feat, DEFAULT_EMB_NUM),
                embedding_dim = emb_dim_dict.get(feat, DEFAULT_EMB_DIM))
            for feat in category_features
        })
        # 2. Bottom
        self.bottom_ppnet = PPNetBlock(
            input_dim=TOTAL_EMB_DIM,
            hidden_dims=bottom_hidden_dims,
            activation="leaky_relu",
            dropout=0.2
        )
        # 3. Tower
        self.click_tower = MultiLayerDNN(
            input_dim=bottom_hidden_dims[-1],
            hidden_dims=tower_hidden_dims,
            output_dim=1,
            activation="leaky_relu",
            dropout=0.2
        )
        self.play_tower = MultiLayerDNN(
            input_dim=bottom_hidden_dims[-1],
            hidden_dims=tower_hidden_dims,
            output_dim=1,
            activation="leaky_relu",
            dropout=0.2
        )


    def forward(self, sample):
        # 1. Embedding lookup
        emb_list, id_emb_list = [], []
        # category features
        for feat in category_features:
            feat_id = sample[feat]
            embed = self.embeddings[feat](feat_id)
            if feat in id_features:
                id_emb_list.append(embed)
            else:
                emb_list.append(embed)
        # dict features
        for key_feat, weight_feat in dict_features.items():
            feat_ids, weight = sample[key_feat], sample[weight_feat]
            shared_emb_feat = dict_features_map[key_feat]
            embeds = self.embeddings[shared_emb_feat](feat_ids)  # shape: [batch_size, len, embed_dim]
            # 加权求和（weight -> [batch_size, len, 1]）
            weighted_embed = (embeds * weight.unsqueeze(-1)).sum(dim=1)  # shape: [batch_size, embed_dim]
            emb_list.append(weighted_embed)
        # seq features (DIN)
        for seq_feat, target_feat in sequence_features_map.items():
            history_ids = sample[seq_feat]          # shape: [batch_size, seq_len]
            target_id = sample[target_feat]         # shape: [batch_size]
            padding_mask = (history_ids == 0)       # mask序列中填充的0
            # 读emb
            target_emb = self.embeddings[target_feat](target_id)    # shape: [batch_size, embed_dim]
            history_embs = self.embeddings[target_feat](history_ids)   # shape: [batch_size, seq_len, embed_dim]
            # 计算 attention
            weights = torch.bmm(history_embs, target_emb.unsqueeze(2))      # shape: [batch_size, seq_len, 1]
            weights = weights.masked_fill(padding_mask.unsqueeze(-1), -1e9)    # mask对应位置的权重
            weights = F.softmax(weights, dim=1)
            # 加权求和
            din_tensor = torch.bmm(history_embs.transpose(1, 2), weights).squeeze(-1)  # shape: [batch_size, embed_dim]
            emb_list.append(din_tensor)
        # 2. ppnet bottom
        normal_embs = torch.cat(emb_list, dim=1)
        id_embs = torch.cat(id_emb_list, dim=1)
        x = self.bottom_ppnet(normal_embs, id_embs)
        # 3. Towers
        click_logit = self.click_tower(x).squeeze(-1)
        play_logit = self.play_tower(x).squeeze(-1)
        return click_logit, play_logit



