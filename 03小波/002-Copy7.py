# -*- coding: utf-8 -*-
"""
EXP-5.0 鲁棒性压力测试 (Renamed Version)
修改点：
1. 图例名称更改为 'Light-TFT' 和 'Wave-Light-TFT'
2. 包含所有之前的 bug 修复
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import tft_core as core

# ================= 配置 =================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 150; seq_len = 32; batch_size = 64
    fc_hidden_dimension = 64; attn_heads = 2; grn_hidden_dim = 64; decoder_hidden_dim = 32; dropout = 0.2
    
    # 补全参数
    wavelet_cache_dir = "wavelet_cache"
    baseline_model_path = "exp4_models/baseline_v2_1.pth"
    ours_model_path = "exp3_5_models/best_dynamic_adaptive.pth"
    result_dir = "results/robustness"
    
    dynamic_window_size = 8
    attention_hidden_dim = 32
    fixed_orig_weight = 0.5
    
    # Ours 配置
    strat_w = "dynamic_adaptive"
    strat_f = "add"
    pos = "middle"
    base = "db4"
    level = 2

DATA_CONFIG = {
    "train": {
        "X_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_train_X_T32.npy",
        "y_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_train_y_T32.npy"
    },
    "val": {
        "X_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_val_X_T32.npy",
        "y_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_val_y_T32.npy"
    },
    "test": {
        "X_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_test_X_T32.npy",
        "y_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_test_y_T32.npy"
    }
}

# ================= 模型定义 =================
class LightTFT_Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        c = Config
        self.proj_x = nn.Sequential(nn.Linear(c.input_dim, c.fc_hidden_dimension), nn.ELU())
        self.lstm = nn.LSTM(c.fc_hidden_dimension, c.fc_hidden_dimension, num_layers=1, batch_first=True)
        self.post_lstm_fc = nn.Sequential(nn.Linear(c.fc_hidden_dimension, c.fc_hidden_dimension), nn.ELU())
        self.attn = nn.MultiheadAttention(c.fc_hidden_dimension, c.attn_heads, batch_first=True)
        self.grn = core.GatedResidualNetwork(c.fc_hidden_dimension, c.grn_hidden_dim, c.fc_hidden_dimension, c.dropout)
        self.decoder = nn.Sequential(nn.Linear(c.fc_hidden_dimension, c.decoder_hidden_dim), nn.ReLU())
        self.head = nn.Linear(c.decoder_hidden_dim, 1)
        self.bn = nn.BatchNorm1d(c.decoder_hidden_dim)
    def forward(self, x):
        curr = self.post_lstm_fc(self.lstm(self.proj_x(x))[0])
        a_out, _ = self.attn(curr, curr, curr)
        out = self.decoder(self.grn(curr + a_out))
        return self.head(self.bn(out[:, -1, :]))

class AddModule(nn.Module):
    def __init__(self, cfg): super().__init__()
    def forward(self, x, w): return x + w
def patched_get_fusion_mod(name, cfg): return AddModule(cfg) if name == "add" else core.SimpleConcatModule(cfg)
core.get_fusion_mod = patched_get_fusion_mod

# ================= 核心测试逻辑 =================
def load_models():
    print("⏳ Loading models...")
    baseline = LightTFT_Baseline().to(Config.device)
    baseline.load_state_dict(torch.load(Config.baseline_model_path, map_location=Config.device))
    baseline.eval()
    
    ours = core.LightTFTv2_1(Config, Config.strat_w, Config.strat_f, Config.pos).to(Config.device)
    ours.load_state_dict(torch.load(Config.ours_model_path, map_location=Config.device))
    ours.eval()
    return baseline, ours

def evaluate(model, loader, noise_level=0.0, mask_rate=0.0, is_ours=False):
    preds, labels = [], []
    with torch.no_grad():
        for x, w, y in loader:
            x, y = x.to(Config.device), y.to(Config.device)
            if is_ours: w = w.to(Config.device)
            
            if noise_level > 0:
                noise_x = torch.randn_like(x) * noise_level
                x = x + noise_x
                if is_ours: w = w + (torch.randn_like(w) * noise_level)
            
            if mask_rate > 0:
                mask_x = torch.rand_like(x) < mask_rate 
                x.masked_fill_(mask_x, 0)
                if is_ours: w.masked_fill_(torch.rand_like(w) < mask_rate, 0)
            
            if is_ours: out = model(x, w)
            else: out = model(x)
            preds.extend(out.cpu().numpy())
            labels.extend(y.cpu().numpy())
            
    metrics = core.calculate_metrics(np.array(labels), np.array(preds))
    return metrics['f1']

# ================= 绘图函数 (已修改名字) =================
def plot_results(levels, scores_base, scores_ours, title, xlabel, filename):
    plt.figure(figsize=(10, 6))
    
    # 【修改点】这里改了 Label
    plt.plot(levels, scores_base, marker='o', linestyle='--', color='gray', 
             linewidth=2, markersize=8, label='Light-TFT') # 原 Baseline
    
    plt.plot(levels, scores_ours, marker='s', linestyle='-', color='#d62728', 
             linewidth=2.5, markersize=8, label='Wave-Light-TFT') # 原 Ours
    
    plt.fill_between(levels, scores_base, scores_ours, color='#d62728', alpha=0.1)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.xticks(levels, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=13, loc='upper right') # 确保在右上角
    
    for i, txt in enumerate(scores_ours):
        plt.annotate(f"{txt:.3f}", (levels[i], scores_ours[i]), 
                     xytext=(0, 10), textcoords='offset points', 
                     ha='center', color='#d62728', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.result_dir, filename), dpi=300)
    print(f"✅ 图表已保存: {filename}")
    plt.close()

# ================= 主流程 =================
def run():
    os.makedirs(Config.result_dir, exist_ok=True)
    print("⏳ Loading data...")
    _, _, ds_te = core.load_data(Config, Config.base, Config.level, DATA_CONFIG)
    loader = DataLoader(ds_te, Config.batch_size, shuffle=False)
    
    baseline, ours = load_models()
    
    # 1. 噪声测试
    print("\n🔥 开始实验 1: 噪声鲁棒性测试...")
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    res_base, res_ours = [], []
    for lvl in noise_levels:
        f1_b = evaluate(baseline, loader, noise_level=lvl, is_ours=False)
        f1_o = evaluate(ours, loader, noise_level=lvl, is_ours=True)
        res_base.append(f1_b)
        res_ours.append(f1_o)
        print(f"   Noise={lvl:<4} | Light-TFT: {f1_b:.4f} | Wave-Light-TFT: {f1_o:.4f}")
    plot_results(noise_levels, res_base, res_ours, "Noise Robustness", "Noise Intensity (std)", "exp5_noise.png")

    # 2. 缺失测试
    print("\n🔥 开始实验 2: 数据缺失容忍度测试...")
    mask_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    res_base, res_ours = [], []
    for rate in mask_rates:
        f1_b = evaluate(baseline, loader, mask_rate=rate, is_ours=False)
        f1_o = evaluate(ours, loader, mask_rate=rate, is_ours=True)
        res_base.append(f1_b)
        res_ours.append(f1_o)
        print(f"   Missing={int(rate*100)}% | Light-TFT: {f1_b:.4f} | Wave-Light-TFT: {f1_o:.4f}")
    plot_results(mask_rates, res_base, res_ours, "Missing Data Tolerance", "Missing Rate", "exp5_missing.png")

if __name__ == "__main__":
    run()