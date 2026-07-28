# -*- coding: utf-8 -*-
"""
文件名: distill_offline_fast.py
作用: 极速离线蒸馏 (Offline Distillation)
特点: 
  1. 不加载 Teacher 模型，直接读取预计算的 Logits
  2. 包含完整的 LightTFT 学生模型定义
  3. 开启 num_workers 多线程加载
  4. 速度优化: 预计 20s/epoch
"""
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, confusion_matrix, accuracy_score
from tqdm import tqdm

# ================= 1. 组件定义 (完整无省略) =================
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

# ================= 2. 学生模型 (完整无省略) =================
class StudentConfig:
    input_dim = 150
    seq_len = 32
    fc_hidden_dimension = 64
    attn_heads = 2
    grn_hidden_dim = 64
    decoder_hidden_dim = 32
    dropout = 0.2

class LightTFT_v2_1_Student(nn.Module):
    def __init__(self):
        super().__init__()
        c = StudentConfig
        # 1. 投影层
        self.proj_x = nn.Sequential(
            nn.Linear(c.input_dim, c.fc_hidden_dimension), 
            nn.ELU()
        )
        # 2. LSTM
        self.lstm = nn.LSTM(c.fc_hidden_dimension, c.fc_hidden_dimension, num_layers=1, batch_first=True)
        self.post_lstm_fc = nn.Sequential(
            nn.Linear(c.fc_hidden_dimension, c.fc_hidden_dimension),
            nn.ELU(),
            nn.Dropout(c.dropout)
        )
        # 3. Attention
        self.attn = nn.MultiheadAttention(c.fc_hidden_dimension, c.attn_heads, dropout=c.dropout, batch_first=True)
        self.grn = GatedResidualNetwork(c.fc_hidden_dimension, c.grn_hidden_dim, c.fc_hidden_dimension, c.dropout)
        # 4. Decoder
        self.decoder = nn.Sequential(
            nn.Linear(c.fc_hidden_dimension, c.decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(c.dropout)
        )
        self.bn = nn.BatchNorm1d(c.decoder_hidden_dim)
        self.head = nn.Linear(c.decoder_hidden_dim, 1)
        
    def forward(self, x):
        curr = self.proj_x(x)
        lstm_out, _ = self.lstm(curr)
        curr = self.post_lstm_fc(lstm_out)
        a_out, _ = self.attn(curr, curr, curr)
        curr = curr + a_out
        curr = self.grn(curr)
        out = self.decoder(curr)
        features = self.bn(out[:, -1, :]) 
        logits = self.head(features)
        return logits

# ================= 3. 实验配置 =================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 路径配置
    BASE_DIR = "/root/autodl-tmp/graduate-thesis/duibi"
    OUTPUT_DIR = os.path.join(BASE_DIR, "distill_offline_results")
    
    # 原始数据
    DATA_ROOT = "/root/autodl-tmp/graduate-thesis/data/tensor_T32"
    DATA_PATHS = {
        "train": {"X": os.path.join(DATA_ROOT, "cic17_W32_S16_train_X_T32.npy"), 
                  "y": os.path.join(DATA_ROOT, "cic17_W32_S16_train_y_T32.npy")},
        "val":   {"X": os.path.join(DATA_ROOT, "cic17_W32_S16_val_X_T32.npy"), 
                  "y": os.path.join(DATA_ROOT, "cic17_W32_S16_val_y_T32.npy")},
        "test":  {"X": os.path.join(DATA_ROOT, "cic17_W32_S16_test_X_T32.npy"), 
                  "y": os.path.join(DATA_ROOT, "cic17_W32_S16_test_y_T32.npy")}
    }
    
    # 预计算的 Logits 路径 (必须与 generate_teacher_logits.py 对应)
    LOGITS_DIR = "/root/autodl-tmp/graduate-thesis/data/distill_logits"
    TEACHER_LOGITS = {
        "train": os.path.join(LOGITS_DIR, "teacher_logits_train.npy"),
        "val":   os.path.join(LOGITS_DIR, "teacher_logits_val.npy")
    }

    # 训练参数
    batch_size = 64
    epochs = 30
    lr = 5e-4
    patience = 8
    num_workers = 4  # 开启多线程加载，提速关键
    
    # 实验组
    KD_EXPERIMENTS = [
        {"T": 3.0, "alpha": 0.5, "name": "Distill_Balanced"},
        {"T": 5.0, "alpha": 0.8, "name": "Distill_SoftFocus"},
        {"T": 2.0, "alpha": 0.2, "name": "Distill_HardFocus"}
    ]

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ================= 4. 工具函数 =================
def calculate_metrics(labels, preds_prob):
    preds_bin = (preds_prob > 0.5).astype(int)
    cm = confusion_matrix(labels, preds_bin)
    fnr = 0
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return {
        "F1": f1_score(labels, preds_bin),
        "Prec": precision_score(labels, preds_bin, zero_division=0),
        "Rec": recall_score(labels, preds_bin, zero_division=0),
        "AUPRC": average_precision_score(labels, preds_prob),
        "FNR": fnr,
        "Acc": accuracy_score(labels, preds_bin)
    }

def distillation_loss_fn(student_logits, teacher_logits, labels, T, alpha, criterion_hard):
    """
    student_logits: (B, 1) 未 Sigmoid
    teacher_logits: (B, 2) 未 Softmax (直接从 numpy 读取)
    """
    # 1. Hard Loss
    L_hard = criterion_hard(student_logits, labels)
    
    # 2. Soft Loss
    # 保持与之前完全一致的逻辑
    with torch.no_grad():
        # 提取 Teacher Class 1 的 Soft 概率
        teacher_probs = F.softmax(teacher_logits / T, dim=1)[:, 1].unsqueeze(1)
    
    # Student Logits / T  <--> Teacher Probs
    L_soft = F.binary_cross_entropy_with_logits(student_logits / T, teacher_probs) * (T * T)
    
    return alpha * L_soft + (1 - alpha) * L_hard

# ================= 5. 训练主循环 =================
def run_experiment(exp_cfg, loaders):
    T = exp_cfg['T']
    alpha = exp_cfg['alpha']
    exp_name = exp_cfg['name']
    
    print(f"\n🧪 [实验启动] {exp_name} | Temp: {T} | Alpha: {alpha}")
    
    student = LightTFT_v2_1_Student().to(Config.device)
    optimizer = optim.Adam(student.parameters(), lr=Config.lr)
    
    pos_weight = torch.tensor([4.0]).to(Config.device)
    criterion_hard = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=False)
    
    best_f1 = 0.0
    patience_cnt = 0
    save_path = os.path.join(Config.OUTPUT_DIR, f"{exp_name}_best.pth")
    
    start_time = time.time()
    
    for ep in range(Config.epochs):
        student.train()
        tr_loss = 0
        
        # 训练: 直接从 loader 拿 teacher logits
        for x, y, t_logits in tqdm(loaders['train'], desc=f"Ep {ep+1}", leave=False):
            x = x.to(Config.device)
            y = y.to(Config.device).unsqueeze(1)
            t_logits = t_logits.to(Config.device)
            
            optimizer.zero_grad()
            s_logits = student(x)
            
            loss = distillation_loss_fn(s_logits, t_logits, y, T, alpha, criterion_hard)
            
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            
        avg_loss = tr_loss / len(loaders['train'])
        
        # 验证
        student.eval()
        preds, targs = [], []
        with torch.no_grad():
            for x, y, _ in loaders['val']: # 验证集也有 logits 但这里不需要计算 loss
                x = x.to(Config.device)
                out = student(x)
                preds.extend(torch.sigmoid(out).cpu().numpy())
                targs.extend(y.cpu().numpy())
        
        metrics = calculate_metrics(np.array(targs), np.array(preds))
        val_f1 = metrics['F1']
        
        epoch_time = time.time() - start_time
        print(f"   Ep {ep+1} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} | Prec: {metrics['Prec']:.4f}")
        
        scheduler.step(val_f1)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(student.state_dict(), save_path)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= Config.patience:
                print("   🛑 触发早停")
                break
                
    train_time = time.time() - start_time
    
    # 最终测试 (Test集没有 Teacher Logits，只评估)
    student.load_state_dict(torch.load(save_path))
    student.eval()
    test_preds, test_targs = [], []
    inf_start = time.time()
    
    with torch.no_grad():
        for x, y in loaders['test']:
            x = x.to(Config.device)
            out = student(x)
            test_preds.extend(torch.sigmoid(out).cpu().numpy())
            test_targs.extend(y.cpu().numpy())
            
    inf_time_ms = (time.time() - inf_start) / len(loaders['test'].dataset) * 1000
    final_metrics = calculate_metrics(np.array(test_targs), np.array(test_preds))
    
    return {
        "Config": exp_name, "T": T, "Alpha": alpha,
        "Train_Time_Total_s": train_time, "Inf_Time_ms": inf_time_ms,
        **final_metrics
    }

# ================= 6. 主程序 =================
def main():
    print("🔥 启动 Offline Knowledge Distillation (Fast Mode)...")
    
    # 1. 加载所有数据 (RAM 足够)
    print("📥 读取数据到内存...")
    X_train = np.load(Config.DATA_PATHS["train"]["X"]).astype(np.float32)
    y_train = np.load(Config.DATA_PATHS["train"]["y"]).astype(np.float32)
    t_train = np.load(Config.TEACHER_LOGITS["train"]).astype(np.float32) # 加载 Teacher Logits
    
    X_val = np.load(Config.DATA_PATHS["val"]["X"]).astype(np.float32)
    y_val = np.load(Config.DATA_PATHS["val"]["y"]).astype(np.float32)
    t_val = np.load(Config.TEACHER_LOGITS["val"]).astype(np.float32)
    
    X_test = np.load(Config.DATA_PATHS["test"]["X"]).astype(np.float32)
    y_test = np.load(Config.DATA_PATHS["test"]["y"]).astype(np.float32)
    
    # 2. 构建 DataLoader
    # 关键：Dataset 返回 (X, Y, TeacherLogits)
    train_set = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(t_train))
    val_set   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val),   torch.from_numpy(t_val))
    test_set  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test)) # 测试集不需要 logits
    
    loaders = {
        "train": DataLoader(train_set, Config.batch_size, shuffle=True,  num_workers=Config.num_workers, pin_memory=True),
        "val":   DataLoader(val_set,   Config.batch_size, shuffle=False, num_workers=Config.num_workers, pin_memory=True),
        "test":  DataLoader(test_set,  Config.batch_size, shuffle=False, num_workers=Config.num_workers, pin_memory=True)
    }
    
    # 3. 运行实验
    all_results = []
    for exp in Config.KD_EXPERIMENTS:
        res = run_experiment(exp, loaders)
        all_results.append(res)
        
    # 4. 打印报告
    print("\n" + "="*90)
    print(f"{'Offline Distillation Final Report':^90}")
    print("="*90)
    header = "{:<18} | {:<5} | {:<5} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}".format(
        "Experiment", "Temp", "Alpha", "F1", "Prec", "Rec", "FNR", "Time(ms)"
    )
    print(header)
    print("-" * 90)
    for r in all_results:
        print("{:<18} | {:<5.1f} | {:<5.1f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f}".format(
            r['Config'], r['T'], r['Alpha'], r['F1'], r['Prec'], r['Rec'], r['FNR'], r['Inf_Time_ms']
        ))
    print("="*90)
    
    with open(os.path.join(Config.OUTPUT_DIR, "kd_fast_summary.json"), 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"📄 结果已保存至 {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()