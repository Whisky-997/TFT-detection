# -*- coding: utf-8 -*-
"""
EXP-4.0 最终基准测试 (The Real Baseline)
目标：测试 Light-TFT v2.1 原生性能 (无小波增强)
架构：Input -> Proj -> [LSTM + FC] -> Attention -> GRN -> Output
"""
import os
import json
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from thop import profile, clever_format
from thop_mha_fix import profile_fixed  # thop MHA 计数修复（Bug fix）
import tft_core as core

# ================= 配置区域 =================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 150; seq_len = 32; batch_size = 64
    
    # Light-TFT v2.1 核心参数 (保持与 Wavelet 版本完全一致)
    fc_hidden_dimension = 64
    attn_heads = 2
    grn_hidden_dim = 64
    decoder_hidden_dim = 32
    dropout = 0.2
    
    epochs = 50; lr = 5e-4; patience = 10
    save_dir = "exp4_models"
    result_path = "results/exp_4_baseline_final.json"

DATA_CONFIG = {
    "train": {"X_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_train_X_T32.npy", "y_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_train_y_T32.npy"},
    "val": {"X_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_val_X_T32.npy", "y_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_val_y_T32.npy"},
    "test": {"X_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_test_X_T32.npy", "y_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_test_y_T32.npy"}
}

# ================= 定义基准模型 (复刻 v2.1 结构，去除小波分支) =================
class LightTFT_v2_1_Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        c = Config
        
        # 1. 投影层
        self.proj_x = nn.Sequential(
            nn.Linear(c.input_dim, c.fc_hidden_dimension), 
            nn.ELU()
        )
        
        # 2. 时序编码器 (v2.1 特征: 1 LSTM + 1 FC)
        self.lstm = nn.LSTM(c.fc_hidden_dimension, c.fc_hidden_dimension, num_layers=1, batch_first=True)
        self.post_lstm_fc = nn.Sequential(
            nn.Linear(c.fc_hidden_dimension, c.fc_hidden_dimension),
            nn.ELU(),
            nn.Dropout(c.dropout)
        )
            
        # 3. TFT 核心组件 (Attention + GRN)
        self.attn = nn.MultiheadAttention(c.fc_hidden_dimension, c.attn_heads, dropout=c.dropout, batch_first=True)
        # 复用 core 中的 GRN 定义
        self.grn = core.GatedResidualNetwork(c.fc_hidden_dimension, c.grn_hidden_dim, c.fc_hidden_dimension, c.dropout)
        
        # 4. 解码器
        self.decoder = nn.Sequential(
            nn.Linear(c.fc_hidden_dimension, c.decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(c.dropout)
        )
        self.head = nn.Linear(c.decoder_hidden_dim, 1)
        self.bn = nn.BatchNorm1d(c.decoder_hidden_dim)
        
    def forward(self, x):
        # A. 投影
        curr = self.proj_x(x)
        
        # B. v2.1 时序编码 (LSTM -> FC)
        lstm_out, _ = self.lstm(curr)
        curr = self.post_lstm_fc(lstm_out)
        
        # C. Attention (Self-Attention)
        a_out, _ = self.attn(curr, curr, curr)
        curr = curr + a_out # 残差 1
        
        # D. GRN
        curr = self.grn(curr) # 内部有残差 2
        
        # E. 输出
        out = self.decoder(curr)
        out = self.bn(out[:, -1, :])
        return self.head(out)

# ================= 数据加载 (仅加载 X, y) =================
def load_data_baseline():
    def _load(s):
        x = torch.from_numpy(np.load(DATA_CONFIG[s]["X_path"])).float()
        y = torch.from_numpy(np.load(DATA_CONFIG[s]["y_path"])).float().unsqueeze(1)
        return TensorDataset(x, y)
    return _load("train"), _load("val"), _load("test")

# ================= 主流程 =================
def run():
    core.create_dirs()
    os.makedirs(Config.save_dir, exist_ok=True)
    
    print(f"🔥 启动 EXP-4.0: Light-TFT v2.1 基准测试 (No Wavelet)")
    
    # 1. 加载数据
    ds_tr, ds_val, ds_te = load_data_baseline()
    ld_tr = DataLoader(ds_tr, Config.batch_size, shuffle=True)
    ld_val = DataLoader(ds_val, Config.batch_size)
    ld_te = DataLoader(ds_te, Config.batch_size)
    
    # 2. 初始化模型
    model = LightTFT_v2_1_Baseline().to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    best_f1 = 0.0
    patience_cnt = 0
    model_save_path = os.path.join(Config.save_dir, "baseline_v2_1.pth")
    
    # 3. 训练循环
    for ep in range(Config.epochs):
        model.train()
        train_loss = 0
        for x, y in ld_tr:
            x, y = x.to(Config.device), y.to(Config.device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in ld_val:
                p = model(x.to(Config.device))
                preds.extend(p.cpu().numpy())
                labels.extend(y.cpu().numpy())
        
        f1 = core.calculate_metrics(np.array(labels), np.array(preds))['f1']
        
        if ep % 5 == 0:
            print(f"   Ep {ep+1}: Loss {train_loss/len(ld_tr):.4f} | Val F1 {f1:.4f}")
        
        scheduler.step(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            patience_cnt = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= Config.patience:
                print(f"   ⏹ Early stop at Ep {ep+1}, Best Val F1: {best_f1:.4f}")
                break
    
    # 4. 测试
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    preds, labels = [], []
    dummy_input = torch.randn(1, 32, 150).to(Config.device)
    
    start = time.time()
    with torch.no_grad():
        for x, y in ld_te:
            p = model(x.to(Config.device))
            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())
    infer_ms = (time.time() - start) * 1000 / len(ds_te)
    
    # 计算所有指标
    p_np = np.array(preds).flatten()
    l_np = np.array(labels).flatten()
    p_bin = (torch.sigmoid(torch.from_numpy(p_np)) > 0.5).numpy().astype(int)
    
    from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
    metrics = {
        "f1": f1_score(l_np, p_bin, average='binary'),
        "prec": precision_score(l_np, p_bin, average='binary'),
        "recall": recall_score(l_np, p_bin, average='binary'),
        "auprc": average_precision_score(l_np, p_np),
        "fnr": 1 - recall_score(l_np, p_bin, average='binary')
    }
    
    # 轻量化指标
    flops, params, _, _ = profile_fixed(model, (dummy_input,), fmt="%.4f")
    
    # 打印结果
    print("\n" + "="*110)
    print(f"{'EXP-4.0 基准测试结果 (Light-TFT v2.1 No Wavelet)':^110}")
    print("="*110)
    headers = ["Model", "F1 Score", "Precision", "Recall", "AUPRC", "FNR", "Params", "FLOPs", "Time(ms)"]
    row_fmt = "| {:<12} | {:<8} | {:<9} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} |"
    
    print(row_fmt.format(*headers))
    print("-" * 110)
    print(row_fmt.format(
        "Baseline", 
        round(metrics['f1'], 4), round(metrics['prec'], 4), round(metrics['recall'], 4), 
        round(metrics['auprc'], 4), round(metrics['fnr'], 4),
        params, flops, round(infer_ms, 3)
    ))
    print("="*110)
    
    # 保存结果
    res = {
        "model": "Light-TFT v2.1 (Baseline)",
        "metrics": metrics,
        "lightweight": {"params": params, "flops": flops, "time": infer_ms}
    }
    with open(Config.result_path, "w") as f:
        json.dump(res, f, indent=4)

if __name__ == "__main__":
    run()