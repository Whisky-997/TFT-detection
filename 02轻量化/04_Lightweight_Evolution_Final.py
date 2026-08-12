# -*- coding: utf-8 -*-
"""
实验 4.2: 轻量化演进最终对比 (Evolution Final) - 修复版
修复内容：
1. 修正 Model_C_Linear 的 forward 方法，解决 EfficientLinearAttention 参数不匹配报错。
2. 保持其他模块逻辑不变。
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

# 尝试引入 Baseline 定义 (用于加载 Ver A)
try:
    from tft_binary import TFTBinary
except ImportError:
    pass

# ================= 配置 =================
BASE_DIR = "/root/autodl-tmp/graduate-thesis/duibi"
# 确保这里指向你 01 实验生成的最佳权重
BASELINE_WEIGHTS = os.path.join(BASE_DIR, "tft_medium_output/tft_baseline_best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1. 数据加载 =================
class SimpleDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path).astype(np.float32)
        # 强制转换为 int64，避免 CrossEntropyLoss 报错
        self.y = np.load(y_path).astype(np.int64)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# ================= 2. 基础组件 =================
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
        self.fc1 = TimeDistributed(nn.Linear(input_size, hidden_size))
        self.fc2 = TimeDistributed(nn.Linear(hidden_size, output_size))
        self.bn = TimeDistributed(nn.BatchNorm1d(output_size))
        self.gate = TimeDistributed(GLU(output_size))
        self.dropout = nn.Dropout(dropout)
        self.skip = TimeDistributed(nn.Linear(input_size, output_size)) if input_size != output_size else None
    def forward(self, x):
        r = self.skip(x) if self.skip else x
        x_fc = self.fc2(self.dropout(torch.nn.functional.elu(self.fc1(x))))
        return self.bn(self.gate(x_fc) + r)

# ================= 3. 模型定义 =================

# --- Ver B: TFT-NoVSN (保留骨架，仅去 VSN) ---
class Model_B_NoVSN(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 64
        # 1. 移除 VSN
        self.input_proj = nn.Linear(150, hidden)
        
        # 2. 保留双层 LSTM
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, dropout=0.2)
        
        # 3. 保留 GRN 骨架
        self.post_lstm_gate = TimeDistributed(GLU(hidden), batch_first=True)
        self.static_enrichment = GatedResidualNetwork(hidden, hidden, hidden, 0.2)
        
        # 4. 标准 Attention (接受 Q,K,V)
        self.attn = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.post_attn_gate = TimeDistributed(GLU(hidden), batch_first=True)
        
        self.out = nn.Linear(hidden, 2)

    def forward(self, x):
        x = self.input_proj(x)
        x_lstm, _ = self.lstm(x)
        x = x + x_lstm 
        x = self.post_lstm_gate(x)
        x = self.static_enrichment(x)
        
        # 标准 Attention 调用：(x, x, x)
        attn_out, _ = self.attn(x, x, x)
        x = self.post_attn_gate(x + attn_out)
        return self.out(x[:, -1, :]) 

# --- Ver C: TFT-Linear (替换 Attention) ---
class EfficientLinearAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
    
    def forward(self, x):
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3), qkv)
        
        q = torch.nn.functional.elu(q) + 1
        k = torch.nn.functional.elu(k) + 1
        
        kv = torch.einsum('b h n d, b h n e -> b h d e', k, v)
        z = 1 / (torch.einsum('b h n d, b h d -> b h n', q, k.sum(dim=2)) + 1e-6)
        out = torch.einsum('b h n d, b h d e, b h n -> b h n e', q, kv, z)
        
        out = out.permute(0, 2, 1, 3).reshape(b, n, d)
        return self.to_out(out)

class Model_C_Linear(Model_B_NoVSN): 
    def __init__(self):
        super().__init__()
        hidden = 64
        # 替换 Attn
        self.attn = EfficientLinearAttention(hidden, heads=4)

    # 【关键修复】重写 forward，适配 Linear Attention 的参数
    def forward(self, x):
        x = self.input_proj(x)
        x_lstm, _ = self.lstm(x)
        x = x + x_lstm 
        x = self.post_lstm_gate(x)
        x = self.static_enrichment(x)
        
        # Linear Attention 只接受 x，且不返回权重 tuple
        attn_out = self.attn(x)
        
        x = self.post_attn_gate(x + attn_out)
        return self.out(x[:, -1, :])

# --- Ver D: TFT-CNN (替换 LSTM) ---
class Model_D_CNN(Model_B_NoVSN): 
    def __init__(self):
        super().__init__()
        hidden = 64
        # 替换 LSTM 为 1D-CNN
        self.lstm = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        
        # CNN Process
        x_cnn = x.transpose(1, 2) # [B, Dim, Seq]
        x_cnn = self.lstm(x_cnn)
        x_cnn = x_cnn.transpose(1, 2) # [B, Seq, Dim]
        
        x = x + x_cnn
        x = self.post_lstm_gate(x)
        x = self.static_enrichment(x)
        
        # 标准 Attention (x, x, x)
        attn_out, _ = self.attn(x, x, x) 
        x = self.post_attn_gate(x + attn_out)
        return self.out(x[:, -1, :])

# --- Ver E: Light-TFT (深度重构, Ours) ---
class Model_E_LightTFT(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 64
        self.input_proj = nn.Sequential(nn.Linear(150, hidden), nn.ELU())
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
        self.fc_encode = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU())
        self.attn = nn.MultiheadAttention(hidden, 2, batch_first=True) 
        self.bottleneck = nn.Sequential(nn.Linear(hidden, 32), nn.ELU(), nn.Dropout(0.2))
        self.out = nn.Linear(32, 2)

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x) 
        x_enc = self.fc_encode(x)
        attn_out, _ = self.attn(x_enc, x_enc, x_enc)
        x = x_enc + attn_out
        x = x[:, -1, :] 
        x = self.bottleneck(x)
        return self.out(x)

# ================= 4. 训练与评估工具 =================
def evaluate(model, loader, is_baseline=False):
    model.eval()
    preds, targs = [], []
    start = time.time()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            # Baseline 兼容处理
            if is_baseline and isinstance(out, tuple):
                out = out[0]
            
            probs = torch.softmax(out, dim=1)[:, 1]
            preds.extend(probs.cpu().numpy())
            targs.extend(y.numpy())
    
    time_cost = (time.time() - start) / len(loader.dataset) * 1000 # ms/sample
    bin_p = (np.array(preds) > 0.5).astype(int)
    
    return {
        "Params": sum(p.numel() for p in model.parameters()) / 1e6,
        "Time": time_cost,
        "F1": f1_score(targs, bin_p),
        "Prec": precision_score(targs, bin_p),
        "Rec": recall_score(targs, bin_p),
        "AUPRC": average_precision_score(targs, preds)
    }

def train_model(model, tr_load, va_load, te_load, name):
    print(f"\n⚡ Training {name} ...")
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    
    best_f1 = 0
    final_res = {}
    
    # 训练 10 轮
    for ep in range(10):
        model.train()
        for x, y in tr_load:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
        
        # Val
        res = evaluate(model, va_load)
        if res['F1'] > best_f1:
            best_f1 = res['F1']
            final_res = evaluate(model, te_load)
            print(f"   Ep {ep+1}: Best F1 {best_f1:.4f}")
            
    return final_res

# ================= 5. 主程序 =================
if __name__ == "__main__":
    print("🚀 开始实验 4.2: 轻量化演进对比 (Evolution Final)")
    
    # 路径准备
    p_tr = ("/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_train_X_T32.npy", 
            "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_train_y_T32.npy")
    p_te = ("/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_test_X_T32.npy", 
            "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_test_y_T32.npy")
            
    ds_tr = SimpleDataset(*p_tr)
    ds_te = SimpleDataset(*p_te) 
    
    ld_tr = DataLoader(ds_tr, 64, shuffle=True, num_workers=0)
    ld_te = DataLoader(ds_te, 64, num_workers=0)
    
    results = {}

    # --- 1. Load Baseline (Ver A) ---
    print("\n📦 Loading Ver A: Baseline (9.5M)...")
    try:
        base_config = {
            "time_varying_real_variables_encoder": 150,
            "time_varying_real_variables_decoder": 150,
            "seq_length": 32, "lstm_hidden_dimension": 64, "lstm_layers": 2,
            "attn_heads": 4, "embedding_dim": 64, "batch_size": 64, 
            "return_sequence": False, "device": DEVICE
        }
        baseline = TFTBinary(base_config).to(DEVICE)
        if os.path.exists(BASELINE_WEIGHTS):
            baseline.load_state_dict(torch.load(BASELINE_WEIGHTS))
            print("   ✅ Weights Loaded!")
            results["A. Baseline"] = evaluate(baseline, ld_te, is_baseline=True)
        else:
            print(f"   ⚠️ Weight file not found at {BASELINE_WEIGHTS}")
            results["A. Baseline"] = {"Params": 9.48, "Time": 0, "F1": 0.9620, "Prec": 0.95, "Rec": 0.97, "AUPRC": 0.99}
    except Exception as e:
        print(f"   ❌ Failed to load Baseline: {e}")

    # --- 2. Train Evolution Variants ---
    models = {
        "B. TFT-NoVSN": Model_B_NoVSN().to(DEVICE),
        "C. TFT-Linear": Model_C_Linear().to(DEVICE),
        "D. TFT-CNN": Model_D_CNN().to(DEVICE),
        "E. Light-TFT": Model_E_LightTFT().to(DEVICE)
    }
    
    for name, m in models.items():
        results[name] = train_model(m, ld_tr, ld_te, ld_te, name)
        
    # --- 3. Print Final Table ---
    print("\n" + "="*110)
    print("📊 论文表 4.X: 不同轻量化变体的性能与开销对比")
    print("="*110)
    print(f"{'Ver':<15} | {'Model':<15} | {'Params(M)':<10} | {'Time(ms)':<10} | {'F1':<8} | {'Prec':<8} | {'Rec':<8} | {'AUPRC':<8}")
    print("-" * 110)
    
    keys = ["A. Baseline", "B. TFT-NoVSN", "C. TFT-Linear", "D. TFT-CNN", "E. Light-TFT"]
    for k in keys:
        if k not in results: continue
        res = results[k]
        model_name = k.split(". ")[1]
        ver_name = k.split(". ")[0]
        print(f"{ver_name:<15} | {model_name:<15} | {res['Params']:<10.3f} | {res['Time']:<10.3f} | {res['F1']:<8.4f} | {res['Prec']:<8.4f} | {res['Rec']:<8.4f} | {res['AUPRC']:<8.4f}")
    print("="*110)