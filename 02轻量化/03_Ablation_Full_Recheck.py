# -*- coding: utf-8 -*-
"""
实验 3.3 (A): 模块重要性直接消融 (Direct Ablation) - 修复版
修复点：
1. 数据加载时强制将标签转换为 int64，解决 TypeError 报错。
2. 确保参数计算和模型结构正确。
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

# ================= 配置 =================
BASE_DIR = "/root/autodl-tmp/graduate-thesis/duibi"
OUTPUT_DIR = os.path.join(BASE_DIR, "ablation_direct")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ⚠️ Baseline 数据 (请根据你 01 实验的实际结果微调)
BASELINE_METRICS = {
    "Params": 9486683,    
    "F1": 0.9620,
    "Prec": 0.9527,
    "Rec": 0.9715,
    "AUPRC": 0.9950
}

# ================= 通用组件 =================
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

# ================= 核心消融模型 =================
class TFT_Ablation_Universal(nn.Module):
    def __init__(self, ablation_mode):
        super().__init__()
        self.mode = ablation_mode
        self.hidden = 64
        self.dropout = 0.2
        self.seq_len = 32
        
        # 1. Input & VSN
        if self.mode == "no_vsn":
            # No-VSN: 直接线性映射，参数量极小
            self.input_proj = nn.Linear(150, self.hidden)
        else:
            # 模拟 VSN: 使用 GRN 保持计算流近似
            self.input_proj = nn.Sequential(
                nn.Linear(150, self.hidden),
                GatedResidualNetwork(self.hidden, self.hidden, self.hidden, self.dropout)
            )

        # 2. Positional Encoding
        if self.mode != "no_pe":
            self.pe = nn.Parameter(torch.randn(1, self.seq_len, self.hidden))
        
        # 3. LSTM Encoder
        if self.mode != "no_lstm":
            self.lstm = nn.LSTM(self.hidden, self.hidden, num_layers=2, batch_first=True, dropout=0.2)
        else:
            self.lstm_replace = nn.Linear(self.hidden, self.hidden)

        # 4. Post-LSTM Gate
        if self.mode != "no_grn":
            self.post_lstm_gate = TimeDistributed(GLU(self.hidden), batch_first=True)
            self.post_lstm_norm = TimeDistributed(nn.BatchNorm1d(self.hidden), batch_first=True)

        # 5. Static Enrichment (GRN)
        if self.mode != "no_grn":
            self.static_enrichment = GatedResidualNetwork(self.hidden, self.hidden, self.hidden, self.dropout)
        
        # 6. Attention
        if self.mode != "no_attn":
            self.attn = nn.MultiheadAttention(self.hidden, 4, batch_first=True)
            self.post_attn_gate = TimeDistributed(GLU(self.hidden), batch_first=True)
            self.post_attn_norm = TimeDistributed(nn.BatchNorm1d(self.hidden), batch_first=True)

        # 7. Output
        self.out_fc = nn.Linear(self.hidden, 2)

    def forward(self, x):
        x = self.input_proj(x)
        
        if self.mode != "no_pe":
            x = x + self.pe[:, :x.size(1), :]

        if self.mode != "no_lstm":
            x_lstm, _ = self.lstm(x)
            x = x + x_lstm
        else:
            x = x + self.lstm_replace(x)

        if self.mode != "no_grn":
            x = self.post_lstm_norm(self.post_lstm_gate(x))

        if self.mode != "no_grn":
            x = self.static_enrichment(x)

        if self.mode != "no_attn":
            attn_out, _ = self.attn(x, x, x)
            if self.mode != "no_grn":
                x = self.post_attn_norm(self.post_attn_gate(x + attn_out))
            else:
                x = x + attn_out

        return self.out_fc(x[:, -1, :])

# ================= 训练流程 =================
class SimpleDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path).astype(np.float32)
        # 【关键修复】加载标签时直接转为 int64 (long)，避免后续类型报错
        self.y = np.load(y_path).astype(np.int64) 
        
    def __len__(self): return len(self.X)
    
    def __getitem__(self, idx): 
        # 这里不需要再指定 dtype=torch.long，因为 self.y 已经是 int64
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def train_eval(mode):
    print(f"\n⚡ 正在测试变体: {mode} ...")
    
    paths = {
        "train": ("/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_train_X_T32.npy", 
                  "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_train_y_T32.npy"),
        "val":   ("/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_val_X_T32.npy", 
                  "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_val_y_T32.npy"),
        "test":  ("/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_test_X_T32.npy", 
                  "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_test_y_T32.npy")
    }
    
    ds_tr = SimpleDataset(*paths["train"])
    ds_va = SimpleDataset(*paths["val"])
    ds_te = SimpleDataset(*paths["test"])
    ld_tr = DataLoader(ds_tr, 64, shuffle=True, num_workers=0)
    ld_va = DataLoader(ds_va, 64, num_workers=0)
    ld_te = DataLoader(ds_te, 64, num_workers=0)
    
    model = TFT_Ablation_Universal(mode).to(DEVICE)
    real_params = sum(p.numel() for p in model.parameters())
    print(f"   📈 真实参数量: {real_params:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_f1 = 0
    patience_cnt = 0
    final_res = {}
    
    for ep in range(20): # 20轮快速验证
        model.train()
        for x, y in ld_tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()
        
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
            if patience_cnt >= 5: break
            
    return final_res

if __name__ == "__main__":
    modes = ["no_vsn", "no_lstm", "no_attn", "no_grn", "no_pe"]
    results = {}
    
    results["Baseline"] = BASELINE_METRICS
    
    for m in modes:
        results[m] = train_eval(m)
        
    print("\n" + "="*100)
    print("📊 直接消融结果 (Direct Ablation)")
    print("="*100)
    print(f"{'Variant':<15} | {'Params':<10} | {'F1':<8} | {'Prec':<8} | {'Rec':<8} | {'AUPRC':<8} | {'Diff':<8}")
    print("-" * 100)
    
    base_f1 = results['Baseline']['F1']
    order = ["Baseline", "no_vsn", "no_pe", "no_grn", "no_attn", "no_lstm"]
    
    for k in order:
        if k not in results: continue
        res = results[k]
        diff = res['F1'] - base_f1
        print(f"{k:<15} | {res['Params']:<10,} | {res['F1']:<8.4f} | {res['Prec']:<8.4f} | {res['Rec']:<8.4f} | {res['AUPRC']:<8.4f} | {diff:+.4f}")
    print("="*100)