# -*- coding: utf-8 -*-
"""
Light-TFT v2.1 (Linear Attention Version)
修改内容：
1. 将 nn.MultiheadAttention 替换为 LinearAttention (基于 Katharopoulos et al. 2020, ELU+1 kernel)
2. 保持所有其他评估指标和流程不变
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
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# 尝试导入 thop 用于计算 FLOPs
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️ 未检测到 thop 库，将跳过 FLOPs 计算 (建议: pip install thop)")

# 导入 tft_core (用于 GRN)
try:
    import tft_core as core
except ImportError:
    # 为了防止报错，如果找不到 tft_core，这里提供一个简单的 Mock 用于测试结构
    print("⚠️ 未找到 tft_core，使用内置 GRN 定义...")
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
    
    # 创建一个 core 命名空间模拟导入
    import types
    core = types.ModuleType('core')
    core.GatedResidualNetwork = GatedResidualNetwork

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 150
    seq_len = 32
    batch_size = 64
    
    # Light-TFT v2.1 核心参数
    fc_hidden_dimension = 64
    attn_heads = 2
    grn_hidden_dim = 64
    decoder_hidden_dim = 32
    dropout = 0.2
    
    epochs = 25
    lr = 5e-4
    patience = 10
    
    # 路径配置
    BASE_DIR = "/root/autodl-tmp/graduate-thesis/duibi/1_17"
    save_dir = os.path.join(BASE_DIR, "saved_models")
    output_dir = os.path.join(BASE_DIR, "lighttft_linear_output")
    
    # 文件路径
    model_save_path = os.path.join(save_dir, "baseline_linear_attn.pth") # 修改文件名以区分
    npz_save_path = os.path.join(output_dir, "results_linear.npz")
    json_save_path = os.path.join(output_dir, "lighttft_linear_metrics.json")

# 确保目录存在
os.makedirs(Config.save_dir, exist_ok=True)
os.makedirs(Config.output_dir, exist_ok=True)
os.makedirs(os.path.join(Config.output_dir, "plots"), exist_ok=True)

# 数据路径
DATA_ROOT = "/root/autodl-tmp/graduate-thesis/data/tensor_T32"
DATA_CONFIG = {
    "train": {
        "X_path": os.path.join(DATA_ROOT, "cic17_W32_S16_train_X_T32.npy"),
        "y_path": os.path.join(DATA_ROOT, "cic17_W32_S16_train_y_T32.npy")
    },
    "val": {
        "X_path": os.path.join(DATA_ROOT, "cic17_W32_S16_val_X_T32.npy"), 
        "y_path": os.path.join(DATA_ROOT, "cic17_W32_S16_val_y_T32.npy")
    },
    "test": {
        "X_path": os.path.join(DATA_ROOT, "cic17_W32_S16_test_X_T32.npy"),
        "y_path": os.path.join(DATA_ROOT, "cic17_W32_S16_test_y_T32.npy")
    }
}

# ================= 新增：线性注意力模块 =================
class LinearAttention(nn.Module):
    """
    基于 Kernel Trick 的线性注意力机制 (Transformers are RNNs, Katharopoulos et al., 2020)
    复杂度: O(N) 而非标准注意力的 O(N^2)
    """
    def __init__(self, embed_dim, heads, dropout=0.1):
        super(LinearAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        assert self.head_dim * heads == embed_dim, "embed_dim 必须能被 heads 整除"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU() # 使用 ELU+1 作为核函数

    def forward(self, x):
        # x: [Batch, Seq, Dim] (假设 Batch First)
        B, N, D = x.shape
        
        # 1. 线性投影并分头
        q = self.q_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2) # [B, H, N, D_h]
        k = self.k_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        
        # 2. 应用核函数 phi(x) = elu(x) + 1，保证非负
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        
        # 3. 线性注意力计算: (Q(K^T V)) / (Q(K^T 1))
        # 先计算 K^T V，这是 O(N) 的关键，结果只有 [B, H, D_h, D_h] 大小，与序列长度 N 无关
        kv = torch.matmul(k.transpose(-2, -1), v) 
        
        # 计算分子: Q * (K^T V) -> [B, H, N, D_h]
        numerator = torch.matmul(q, kv)
        
        # 计算分母: Q * (sum(K)^T)
        k_sum = k.sum(dim=-2, keepdim=True).transpose(-2, -1) # [B, H, D_h, 1]
        denominator = torch.matmul(q, k_sum) # [B, H, N, 1]
        
        # 归一化
        attn_out = numerator / (denominator + 1e-6)
        
        # 4. 恢复维度
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)
        output = self.out_proj(attn_out)
        
        return self.dropout(output)

# ================= 模型定义 (修改版) =================
class LightTFT_v2_1_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        c = Config
        
        # 1. 投影层
        self.proj_x = nn.Sequential(
            nn.Linear(c.input_dim, c.fc_hidden_dimension), 
            nn.ELU()
        )
        
        # 2. 时序编码器 (LSTM + FC)
        self.lstm = nn.LSTM(c.fc_hidden_dimension, c.fc_hidden_dimension, num_layers=1, batch_first=True)
        self.post_lstm_fc = nn.Sequential(
            nn.Linear(c.fc_hidden_dimension, c.fc_hidden_dimension),
            nn.ELU(),
            nn.Dropout(c.dropout)
        )
            
        # 3. TFT 核心组件 - 【替换为线性注意力】
        # 原版: self.attn = nn.MultiheadAttention(...)
        print("✨ 初始化: 使用 Linear Attention 替代标准 Multihead Attention")
        self.attn = LinearAttention(c.fc_hidden_dimension, c.attn_heads, dropout=c.dropout)
        
        self.grn = core.GatedResidualNetwork(c.fc_hidden_dimension, c.grn_hidden_dim, c.fc_hidden_dimension, c.dropout)
        
        # 4. 解码器
        self.decoder = nn.Sequential(
            nn.Linear(c.fc_hidden_dimension, c.decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(c.dropout)
        )
        # BatchNorm
        self.bn = nn.BatchNorm1d(c.decoder_hidden_dim)
        # Head
        self.head = nn.Linear(c.decoder_hidden_dim, 1)
        
    def forward(self, x, return_features=False):
        # A. 投影
        curr = self.proj_x(x)
        
        # B. LSTM
        lstm_out, _ = self.lstm(curr)
        curr = self.post_lstm_fc(lstm_out)
        
        # C. Attention (Linear)
        # 线性注意力不需要传入 (curr, curr, curr)，只需要传入 x 即可进行 Self-Attention
        a_out = self.attn(curr)
        curr = curr + a_out
        
        # D. GRN
        curr = self.grn(curr)
        
        # E. 解码
        out = self.decoder(curr)
        
        # 取最后一个时间步 + BatchNorm作为特征
        features = self.bn(out[:, -1, :]) 
        
        logits = self.head(features)
        
        if return_features:
            return logits, features
        return logits

# ================= 数据加载 =================
def load_data_baseline():
    def _load(s):
        x = torch.from_numpy(np.load(DATA_CONFIG[s]["X_path"])).float()
        y = torch.from_numpy(np.load(DATA_CONFIG[s]["y_path"])).float().unsqueeze(1)
        return TensorDataset(x, y)
    return _load("train"), _load("val"), _load("test")

# ================= 工具函数 =================
def calculate_metrics(labels, preds_prob, threshold=0.5):
    preds_bin = (preds_prob > threshold).astype(int)
    cm = confusion_matrix(labels, preds_bin)
    
    fnr = 0
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
    return {
        "Accuracy": accuracy_score(labels, preds_bin),
        "F1-Score": f1_score(labels, preds_bin, average='binary'),
        "Precision": precision_score(labels, preds_bin, average='binary'),
        "Recall": recall_score(labels, preds_bin, average='binary'),
        "AUPRC": average_precision_score(labels, preds_prob),
        "FNR": fnr,
        "Confusion_Matrix": cm.tolist()
    }

def get_lightweight_metrics(model, device):
    """计算 FLOPs 和参数量"""
    if not THOP_AVAILABLE:
        return "N/A", "N/A"
    
    # 创建 Dummy Input [Batch=1, Seq, Dim]
    dummy_input = torch.randn(1, Config.seq_len, Config.input_dim).to(device)
    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.3f")
        return flops_str, params_str
    except Exception as e:
        print(f"FLOPs calculation failed: {e}")
        return "Error", "Error"

def plot_results(train_losses, val_losses, metrics, features, labels, output_dir):
    plot_path = os.path.join(output_dir, "plots")
    
    if train_losses and val_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Val')
        plt.title('LightTFT Linear Attn Loss')
        plt.legend()
        plt.savefig(os.path.join(plot_path, 'loss.png'))
        plt.close()
    
    cm = np.array(metrics['Confusion_Matrix'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title('LightTFT Linear Attn CM')
    plt.savefig(os.path.join(plot_path, 'cm.png'))
    plt.close()
    
    print("🎨 Generating t-SNE plot...")
    if len(features) > 10000:
        idx = np.random.choice(len(features), 10000, replace=False)
        features_sub = features[idx]
        labels_sub = labels[idx]
    else:
        features_sub = features
        labels_sub = labels
        
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=40, init='pca')
        f_2d = tsne.fit_transform(features_sub)
        plt.figure(figsize=(10, 10))
        plt.scatter(f_2d[labels_sub==0,0], f_2d[labels_sub==0,1], c='dodgerblue', alpha=0.4, s=15, label='Normal', edgecolors='w', linewidth=0.1)
        plt.scatter(f_2d[labels_sub==1,0], f_2d[labels_sub==1,1], c='crimson', alpha=0.4, s=15, label='Attack', edgecolors='w', linewidth=0.1)
        plt.title('LightTFT Linear Attn t-SNE')
        plt.legend()
        plt.savefig(os.path.join(plot_path, 'tsne.png'))
        plt.close()
        print("✅ t-SNE saved.")
    except Exception as e:
        print(f"❌ t-SNE Error: {e}")

# ================= 主流程 =================
def run():
    print(f"🔥 启动 Light-TFT (Linear Attention) 评估流程")
    print(f"   输出目录: {Config.output_dir}")
    
    # 尝试加载数据
    try:
        ds_tr, ds_val, ds_te = load_data_baseline()
        ld_tr = DataLoader(ds_tr, Config.batch_size, shuffle=True, num_workers=0)
        ld_val = DataLoader(ds_val, Config.batch_size, num_workers=0)
        ld_te = DataLoader(ds_te, Config.batch_size, num_workers=0)
    except FileNotFoundError:
        print("❌ 错误: 找不到数据文件。请检查 DATA_ROOT 路径。")
        print(f"   当前配置路径: {DATA_ROOT}")
        return

    # 1. 初始化模型 (使用新的线性版本)
    model = LightTFT_v2_1_Linear().to(Config.device)
    
    # 2. 训练或加载
    train_losses, val_losses = [], []
    if os.path.exists(Config.model_save_path):
        print(f"📂 检测到现有权重，直接加载: {Config.model_save_path}")
        model.load_state_dict(torch.load(Config.model_save_path))
    else:
        print(f"⚙️ 未找到权重，开始训练...")
        optimizer = optim.Adam(model.parameters(), lr=Config.lr)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        best_f1 = 0.0; patience_cnt = 0
        
        for ep in range(Config.epochs):
            model.train()
            e_loss = 0
            for x, y in tqdm(ld_tr, desc=f"Ep {ep+1}"):
                x, y = x.to(Config.device), y.to(Config.device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                e_loss += loss.item()
            train_losses.append(e_loss/len(ld_tr))
            
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for x, y in ld_val:
                    out = model(x.to(Config.device))
                    preds.extend(torch.sigmoid(out).cpu().numpy())
                    labels.extend(y.cpu().numpy())
            
            f1 = f1_score(np.array(labels), (np.array(preds)>0.5).astype(int))
            print(f"   Ep {ep+1} | Val F1: {f1:.4f}")
            val_losses.append(0)
            
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), Config.model_save_path)
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= Config.patience: break
            scheduler.step(f1)
        # 加载最佳模型
        if os.path.exists(Config.model_save_path):
            model.load_state_dict(torch.load(Config.model_save_path))

    # 3. 计算 FLOPs
    print("💻 Calculating FLOPs...")
    temp_model = LightTFT_v2_1_Linear().to(Config.device)
    flops_str, params_str = get_lightweight_metrics(temp_model, Config.device)
    del temp_model
    print(f"💻 Model Stats -> FLOPs: {flops_str} | Params: {params_str}")

    # 4. 最终测试
    print("🧪 开始最终测试与特征提取...")
    model.eval()
    t_probs, t_labels, t_features = [], [], []
    start_time = time.time()
    
    with torch.no_grad():
        for x, y in tqdm(ld_te, desc="Testing"):
            x = x.to(Config.device)
            out, feats = model(x, return_features=True)
            
            t_probs.extend(torch.sigmoid(out).cpu().numpy())
            t_labels.extend(y.numpy())
            t_features.extend(feats.cpu().numpy())
            
    inf_time_sec = (time.time() - start_time) / len(ds_te)
    inf_time_ms = inf_time_sec * 1000
    
    t_probs = np.array(t_probs).flatten()
    t_labels = np.array(t_labels).flatten()
    t_features = np.array(t_features)
    
    metrics = calculate_metrics(t_labels, t_probs)
    
    final_results = {
        **metrics,
        "FLOPs": flops_str,
        "Params": params_str,
        "Inference_Time_ms": inf_time_ms,
        "Inference_Time_sec": inf_time_sec
    }
    
    print("\n" + "="*50)
    print(f"{'Light-TFT (Linear) Final Results':^50}")
    print("="*50)
    print(f"Accuracy:  {final_results['Accuracy']:.4f}")
    print(f"F1 Score:  {final_results['F1-Score']:.4f}")
    print(f"Precision: {final_results['Precision']:.4f}")
    print(f"Recall:    {final_results['Recall']:.4f}")
    print(f"AUPRC:     {final_results['AUPRC']:.4f}")
    print("-" * 50)
    print(f"FLOPs:     {final_results['FLOPs']}")
    print(f"Params:    {final_results['Params']}")
    print(f"Inf Time:  {final_results['Inference_Time_ms']:.4f} ms/sample")
    print("="*50)
    
    with open(Config.json_save_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"📄 完整指标已保存至 JSON: {Config.json_save_path}")
    
    np.savez(Config.npz_save_path, probs=t_probs, labels=t_labels, features=t_features, inf_time=inf_time_sec)
    
    plot_results(train_losses, val_losses, metrics, t_features, t_labels, Config.output_dir)

if __name__ == "__main__":
    run()