# -*- coding: utf-8 -*-
"""
文件名: tft_core.py
作用: Wavelet-LightTFT v2.1 核心库 (修复 Attention 维度报错版)
"""
import torch
import torch.nn as nn
import numpy as np
import os
import pywt
import logging
import json
from tqdm import tqdm
from thop import profile, clever_format
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

# ================= 0. 辅助工具 =================
def create_dirs(base_dir="."):
    """创建实验所需的标准目录结构"""
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "wavelet_cache"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "lighttft_final_models"), exist_ok=True)

# ================= 1. 基础组件 (GLU, GRN, TimeDistributed) =================
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first
    def forward(self, x):
        if len(x.size()) <= 2: return self.module(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)
        if self.batch_first: y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else: y = y.view(-1, x.size(1), y.size(-1))
        return y

class GLU(nn.Module):
    def __init__(self, input_size):
        super(GLU, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return torch.mul(self.sigmoid(self.fc1(x)), self.fc2(x))

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_state_size, output_size, dropout, batch_first=True):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.skip_layer = TimeDistributed(nn.Linear(input_size, output_size)) if input_size != output_size else None
        self.fc1 = TimeDistributed(nn.Linear(input_size, hidden_state_size), batch_first=batch_first)
        self.elu1 = nn.ELU()
        self.fc2 = TimeDistributed(nn.Linear(hidden_state_size, output_size), batch_first=batch_first)
        self.dropout_layer = nn.Dropout(dropout)
        self.bn = TimeDistributed(nn.BatchNorm1d(output_size), batch_first=batch_first)
        self.gate = TimeDistributed(GLU(output_size), batch_first=batch_first)
    def forward(self, x):
        residual = self.skip_layer(x) if self.skip_layer is not None else x
        x = self.fc1(x)
        x = self.elu1(x)
        x = self.fc2(x)
        x = self.dropout_layer(x)
        x = self.gate(x)
        x = x + residual
        x = self.bn(x)
        return x

# ================= 2. 策略组件 (加权 & 融合) =================
class FixedWeightModule(nn.Module):
    def __init__(self, cfg): super().__init__(); self.w = cfg.fixed_orig_weight
    def forward(self, x, w): return x * self.w, w * (1 - self.w)

class DynamicWeightModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_win = cfg.seq_len // cfg.dynamic_window_size
        self.w_win = nn.Parameter(torch.zeros(self.num_win)) 
        self.win_size = cfg.dynamic_window_size
    def forward(self, x, w):
        win_w = torch.sigmoid(self.w_win).repeat_interleave(self.win_size)
        win_w = win_w.view(1, -1, 1).to(x.device)
        return x * win_w, w * (1.0 - win_w)

class SimpleConcatModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.fc_hidden_dimension * 2, cfg.fc_hidden_dimension)
        self.ln = nn.LayerNorm(cfg.fc_hidden_dimension)
        self.relu = nn.ReLU()
    def forward(self, x, w):
        return self.relu(self.ln(self.fc(torch.cat([x, w], dim=-1))))

# 【修复点】LightAttentionModule
class LightAttentionModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, a = cfg.fc_hidden_dimension, cfg.attention_hidden_dim
        self.q = nn.Linear(h, a); self.k = nn.Linear(h, a); self.v = nn.Linear(h, a)
        self.out = nn.Linear(a, h); self.ln = nn.LayerNorm(h); self.relu = nn.ReLU()
    def forward(self, x, w):
        stack = torch.stack([x, w], dim=-2)
        q = self.q(stack); k = self.k(stack); v = self.v(stack)
        # 【修复】使用 self.q.out_features 获取维度，避免 NameError
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.q.out_features**0.5)
        fused = torch.mean(torch.matmul(torch.softmax(scores, dim=-1), v), dim=-2)
        return self.relu(self.ln(self.out(fused)))

def get_weight_mod(name, cfg): return DynamicWeightModule(cfg) if name == "dynamic_adaptive" else FixedWeightModule(cfg)
def get_fusion_mod(name, cfg): return LightAttentionModule(cfg) if name == "light_attention" else SimpleConcatModule(cfg)

# ================= 3. 核心模型: Light-TFT v2.1 =================
class LightTFTv2_1(nn.Module):
    def __init__(self, config, weight_strat, fusion_strat, fusion_pos):
        super().__init__()
        c = config
        self.fusion_pos = fusion_pos
        self.use_residual = True
        
        self.proj_x = nn.Sequential(nn.Linear(c.input_dim, c.fc_hidden_dimension), nn.ELU())
        self.proj_w = nn.Sequential(nn.Linear(c.input_dim, c.fc_hidden_dimension), nn.ELU())
        self.w_mod = get_weight_mod(weight_strat, c)
        self.f_mod = get_fusion_mod(fusion_strat, c)
        
        self.lstm = nn.LSTM(c.fc_hidden_dimension, c.fc_hidden_dimension, num_layers=1, batch_first=True)
        self.post_lstm_fc = nn.Sequential(nn.Linear(c.fc_hidden_dimension, c.fc_hidden_dimension), nn.ELU(), nn.Dropout(c.dropout))
        
        self.attn = nn.MultiheadAttention(c.fc_hidden_dimension, c.attn_heads, dropout=c.dropout, batch_first=True)
        self.grn = GatedResidualNetwork(c.fc_hidden_dimension, c.grn_hidden_dim, c.fc_hidden_dimension, c.dropout)
        self.decoder = nn.Sequential(nn.Linear(c.fc_hidden_dimension, c.decoder_hidden_dim), nn.ReLU(), nn.Dropout(c.dropout))
        self.head = nn.Linear(c.decoder_hidden_dim, 1)
        self.bn = nn.BatchNorm1d(c.decoder_hidden_dim)
        
        self.to(c.device)

    def forward(self, x, w):
        hx = self.proj_x(x); hw = self.proj_w(w)
        curr = hx
        if self.fusion_pos == "early":
            wx, ww = self.w_mod(hx, hw)
            fused = self.f_mod(wx, ww)
            if self.use_residual: fused = fused + hx
            curr = fused
            
        lstm_out, _ = self.lstm(curr)
        curr = self.post_lstm_fc(lstm_out)
        
        if self.fusion_pos == "middle":
            wx, ww = self.w_mod(curr, hw) 
            fused = self.f_mod(wx, ww)
            if self.use_residual: fused = fused + curr
            curr = fused

        a_out, _ = self.attn(curr, curr, curr)
        curr = self.grn(curr + a_out)
        out = self.decoder(curr)
        return self.head(self.bn(out[:, -1, :]))

# ================= 4. 工具函数 =================
def load_data(cfg, base, level, data_config):
    lvl_str = str(level) if isinstance(level, int) else "_".join(map(str, level))
    cache_name = f"cic17_{base}_{lvl_str}"
    
    def get_w(x, split):
        path = os.path.join(cfg.wavelet_cache_dir, f"{cache_name}_{split}.npz")
        if os.path.exists(path): return torch.from_numpy(np.load(path)["data"]).float()
        
        x_np = x.reshape(-1, cfg.input_dim).cpu().numpy()
        res = []
        levels = level if isinstance(level, list) else [level]
        for item in tqdm(x_np, desc=f"Wavelet {split}"):
            c_list = []
            for l in levels:
                c = np.concatenate(pywt.wavedec(item, base, level=l))
                if len(c) < cfg.input_dim: c = np.pad(c, (0, cfg.input_dim-len(c)))
                c_list.append(c[:cfg.input_dim])
            res.append(np.mean(c_list, axis=0)) 
        data = np.array(res).reshape(x.shape[0], cfg.seq_len, cfg.input_dim)
        np.savez(path, data=data)
        return torch.from_numpy(data).float()

    def _load(s):
        x = torch.from_numpy(np.load(data_config[s]["X_path"])).float()
        y = torch.from_numpy(np.load(data_config[s]["y_path"])).float().unsqueeze(1)
        w = get_w(x, s)
        return TensorDataset(x, w, y)
    return _load("train"), _load("val"), _load("test")

def calculate_metrics(labels, preds):
    if isinstance(preds, torch.Tensor): preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
    
    p_bin = (torch.sigmoid(torch.from_numpy(preds)) > 0.5).numpy().astype(int)
    labels = labels.flatten()
    return {
        "f1": f1_score(labels, p_bin, average='binary'),
        "fnr": 1 - recall_score(labels, p_bin, average='binary'),
        "prec": precision_score(labels, p_bin, average='binary'),
        "recall": recall_score(labels, p_bin, average='binary')
    }