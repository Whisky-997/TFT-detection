# -*- coding: utf-8 -*-
"""
实验 3.3 (B): 分层消融实验 (Hierarchical Ablation) - 修复版
逻辑：
1. 设定 Base 模型为 "No-VSN" (去掉了变量选择网络的 TFT)。
2. 在 Base 的基础上，进一步去掉 LSTM、Attention 或 GRN。
3. 目的：验证即使没有 VSN，剩下的骨架组件是否依然有效。

修复内容：
- 数据加载强制转换为 int64，修复 TypeError。
- 实时计算模型参数量。
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

# ================= 1. 配置 =================
BASE_DIR = "/root/autodl-tmp/graduate-thesis/duibi"
OUTPUT_DIR = os.path.join(BASE_DIR, "ablation_hierarchical")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据路径配置
DATA_PATHS = {
    "train": ("/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_train_X_T32.npy", 
              "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_train_y_T32.npy"),
    "val":   ("/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_val_X_T32.npy", 
              "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_val_y_T32.npy"),
    "test":  ("/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_test_X_T32.npy", 
              "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_test_y_T32.npy")
}

# ================= 2. 核心组件 (Building Blocks) =================
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super().__init__()
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
        super().__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return torch.mul(self.sigmoid(self.fc1(x)), self.fc2(x))

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.skip_layer = TimeDistributed(nn.Linear(input_size, output_size)) if input_size != output_size else None
        self.fc1 = TimeDistributed(nn.Linear(input_size, hidden_size))
        self.fc2 = TimeDistributed(nn.Linear(hidden_size, output_size))
        self.dropout = nn.Dropout(dropout)
        self.bn = TimeDistributed(nn.BatchNorm1d(output_size))
        self.gate = TimeDistributed(GLU(output_size))
    def forward(self, x):
        residual = self.skip_layer(x) if self.skip_layer else x
        x_fc = self.fc2(self.dropout(torch.nn.functional.elu(self.fc1(x))))
        return self.bn(self.gate(x_fc) + residual)

# ================= 3. 分层消融模型架构 =================
class TFT_Hierarchical(nn.Module):
    def __init__(self, tags):
        super().__init__()
        self.tags = tags # e.g., ["no_vsn", "no_lstm"]
        self.hidden = 64
        self.dropout = 0.2
        self.seq_len = 32
        
        # 1. Input: 既然是分层消融，Base 就是 No-VSN，所以这里固定没有 VSN
        # 使用简单的线性映射代替复杂的 VSN
        self.input_proj = nn.Linear(150, self.hidden)

        # 2. LSTM (可选关闭)
        if "no_lstm" not in self.tags:
            self.lstm = nn.LSTM(self.hidden, self.hidden, num_layers=2, batch_first=True, dropout=0.2)
        else:
            # 如果没有 LSTM，用线性层保持维度，但不具备时序记忆
            self.lstm_replace = nn.Linear(self.hidden, self.hidden)

        # 3. Post-LSTM Gate (可选关闭)
        if "no_grn" not in self.tags:
            self.post_lstm_gate = TimeDistributed(GLU(self.hidden), batch_first=True)
            self.post_lstm_norm = TimeDistributed(nn.BatchNorm1d(self.hidden), batch_first=True)
        
        # 4. Attention (可选关闭)
        if "no_attn" not in self.tags:
            self.attn = nn.MultiheadAttention(self.hidden, 4, batch_first=True)
            self.post_attn_gate = TimeDistributed(GLU(self.hidden), batch_first=True)
            self.post_attn_norm = TimeDistributed(nn.BatchNorm1d(self.hidden), batch_first=True)

        # 5. Output
        self.out = nn.Linear(self.hidden, 2)

    def forward(self, x):
        # Input
        x = self.input_proj(x)
        
        # LSTM Block
        if "no_lstm" not in self.tags:
            x_lstm, _ = self.lstm(x)
            x = x + x_lstm # 残差连接
        else:
            x = x + self.lstm_replace(x)

        # LSTM Gate
        if "no_grn" not in self.tags:
            x = self.post_lstm_norm(self.post_lstm_gate(x))

        # Attention Block
        if "no_attn" not in self.tags:
            attn_out, _ = self.attn(x, x, x)
            # 注意：如果 no_grn，我们也去掉了 Attn 后面的 Gate/Norm
            if "no_grn" not in self.tags:
                x = self.post_attn_norm(self.post_attn_gate(x + attn_out))
            else:
                x = x + attn_out 
        
        # Output (取最后一个时间步)
        return self.out(x[:, -1, :])

# ================= 4. 数据加载与训练 =================
class SimpleDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path).astype(np.float32)
        # 【关键修复】直接在这里转为 int64，防止后续报错
        self.y = np.load(y_path).astype(np.int64)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): 
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def train_eval(exp_name, tags):
    print(f"\n⚡ 正在测试分层变体: {exp_name}")
    print(f"   Tags: {tags}")
    
    # 加载数据
    ld_tr = DataLoader(SimpleDataset(*DATA_PATHS["train"]), 64, shuffle=True, num_workers=0)
    ld_va = DataLoader(SimpleDataset(*DATA_PATHS["val"]), 64, num_workers=0)
    ld_te = DataLoader(SimpleDataset(*DATA_PATHS["test"]), 64, num_workers=0)
    
    # 初始化模型
    model = TFT_Hierarchical(tags).to(DEVICE)
    real_params = sum(p.numel() for p in model.parameters())
    print(f"   📈 真实参数量: {real_params:,}")
    
    # 优化器配置
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_f1 = 0
    patience_cnt = 0
    final_res = {}
    
    # 训练循环 (20 Epochs)
    for ep in range(20):
        model.train()
        for x, y in ld_tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()
        
        # 验证
        model.eval()
        preds, targs = [], []
        with torch.no_grad():
            for x, y in ld_va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                probs = torch.softmax(model(x), dim=1)[:, 1]
                preds.extend(probs.cpu().numpy()); targs.extend(y.cpu().numpy())
        
        f1 = f1_score(targs, (np.array(preds) > 0.5).astype(int))
        scheduler.step(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            patience_cnt = 0
            # 立即测试
            model.eval()
            te_p, te_t = [], []
            with torch.no_grad():
                for x, y in ld_te:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    probs = torch.softmax(model(x), dim=1)[:, 1]
                    te_p.extend(probs.cpu().numpy()); te_t.extend(y.cpu().numpy())
            
            bin_p = (np.array(te_p) > 0.5).astype(int)
            final_res = {
                "Params": real_params,
                "F1": f1_score(te_t, bin_p),
                "Prec": precision_score(te_t, bin_p),
                "Rec": recall_score(te_t, bin_p),
                "AUPRC": average_precision_score(te_t, te_p)
            }
            print(f"   Ep {ep+1} | Val F1: {f1:.4f} | Test F1: {final_res['F1']:.4f} (Best)")
        else:
            patience_cnt += 1
            if patience_cnt >= 5: break # 早停
            
    return final_res

# ================= 5. 主程序入口 =================
if __name__ == "__main__":
    # 定义实验组
    # 注意：所有组都默认隐含 "no_vsn"，因为这是分层实验的基础
    experiments = {
        "A. Base (No-VSN)":   ["no_vsn"],
        "B. w/o LSTM":        ["no_vsn", "no_lstm"],
        "C. w/o Attention":   ["no_vsn", "no_attn"],
        "D. w/o GRN":         ["no_vsn", "no_grn"]
    }
    
    results = {}
    for name, tags in experiments.items():
        results[name] = train_eval(name, tags)
        
    print("\n" + "="*110)
    print("📊 分层消融最终结果 (Hierarchical Ablation Results)")
    print("="*110)
    print(f"{'Variant':<25} | {'Params':<12} | {'F1-Score':<10} | {'Prec':<10} | {'Recall':<10} | {'AUPRC':<10} | {'Drop':<8}")
    print("-" * 110)
    
    # 计算相对于 Base (A组) 的下降幅度
    base_f1 = results["A. Base (No-VSN)"]["F1"]
    
    for name, res in results.items():
        diff = res["F1"] - base_f1
        print(f"{name:<25} | {res['Params']:<12,} | {res['F1']:<10.4f} | {res['Prec']:<10.4f} | {res['Rec']:<10.4f} | {res['AUPRC']:<10.4f} | {diff:+.4f}")
    print("="*110)