# -*- coding: utf-8 -*-
"""
WaveLightTFT (Dynamic) Final Evaluation Script (With Feature Hook)
修改内容：
1. 新增 FLOPs 和 Params 计算 (使用 thop)。
2. 新增 Accuracy 指标。
3. 输出完整结果到 JSON 文件。
4. 保持 Hook 特征提取与绘图功能。
"""
import os
import json
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# 尝试导入 thop
try:
    from thop import profile, clever_format
    from thop_mha_fix import profile_fixed  # thop MHA 计数修复（Bug fix）
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️ 未检测到 thop 库，将跳过 FLOPs 计算 (建议: pip install thop)")

# 导入核心库
try:
    import tft_core as core
except ImportError:
    raise ImportError("❌ 缺少 tft_core.py！请将 tft_core.py 复制到当前脚本所在目录。")

warnings.filterwarnings('ignore')

# ================= 1. 动态注入 Add 策略 =================
class AddModule(nn.Module):
    def __init__(self, cfg): super().__init__()
    def forward(self, x, w): return x + w

def patched_get_fusion_mod(name, cfg):
    if name == "simple_concat": return core.SimpleConcatModule(cfg)
    if name == "light_attention": return core.LightAttentionModule(cfg)
    if name == "add": return AddModule(cfg)
    raise ValueError(f"Unknown fusion: {name}")

core.get_fusion_mod = patched_get_fusion_mod
print("✅ 已动态注入 Add 融合策略支持")

# ================= 2. 配置区域 =================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 基础参数
    input_dim = 150
    seq_len = 32
    batch_size = 64
    
    # 模型超参 (最优配置)
    fc_hidden_dimension = 64
    attn_heads = 2
    grn_hidden_dim = 64
    decoder_hidden_dim = 32
    dropout = 0.2
    
    # 策略参数 (Dynamic)
    dynamic_window_size = 8
    attention_hidden_dim = 32
    fixed_orig_weight = 0.5
    
    # 训练参数
    epochs = 60
    lr = 5e-4
    patience = 15
    
    # 路径配置
    BASE_DIR = "/root/autodl-tmp/graduate-thesis/duibi/1_17"
    wavelet_cache_dir = "/root/autodl-tmp/graduate-thesis/duibi/1_17"
    
    save_dir = os.path.join(BASE_DIR, "saved_models")
    output_dir = os.path.join(BASE_DIR, "wavelighttft_output")
    
    model_save_path = os.path.join(save_dir, "best_wavelighttft_dynamic.pth")
    npz_save_path = os.path.join(output_dir, "results.npz")
    json_save_path = os.path.join(output_dir, "wavelighttft_metrics.json") # 新增 JSON 路径

# 确保目录存在
os.makedirs(Config.save_dir, exist_ok=True)
os.makedirs(Config.output_dir, exist_ok=True)
os.makedirs(os.path.join(Config.output_dir, "plots"), exist_ok=True)
os.makedirs(Config.wavelet_cache_dir, exist_ok=True)

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

# ================= 3. 特征提取 Hook =================
extracted_features = []

def feature_hook(module, input, output):
    extracted_features.append(input[0].detach().cpu())

def register_feature_hook(model):
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append(module)
    
    if len(linear_layers) > 0:
        last_layer = linear_layers[-1]
        last_layer.register_forward_hook(feature_hook)
        print(f"✅ 已成功挂载特征提取 Hook 到层: {last_layer}")
    else:
        print("⚠️ 警告: 未找到 Linear 层，特征提取可能失败！")

# ================= 4. 工具函数 =================

def calculate_metrics(labels, preds_prob, threshold=0.5):
    preds_bin = (preds_prob > threshold).astype(int)
    cm = confusion_matrix(labels, preds_bin)
    fnr = 0
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return {
        "Accuracy": accuracy_score(labels, preds_bin), # 新增
        "F1-Score": f1_score(labels, preds_bin, average='binary'),
        "Precision": precision_score(labels, preds_bin, average='binary'),
        "Recall": recall_score(labels, preds_bin, average='binary'),
        "AUPRC": average_precision_score(labels, preds_prob),
        "FNR": fnr,
        "Confusion_Matrix": cm.tolist() # 转为 list
    }

def get_lightweight_metrics(config, strat, best_strat, best_pos, device):
    """
    使用临时模型计算 FLOPs / Params，避免污染主模型。

    用 thop_mha_fix.profile_fixed 统计：FLOPs 含 nn.MultiheadAttention（thop 0.1.1
    原本漏算 MHA），Params 取 sum(p.numel()) 真实值（避免漏算 MHA.in_proj / w_win 等
    裸 Parameter）。
    """
    if not THOP_AVAILABLE:
        return "N/A", "N/A"

    print("💻 Creating Temp Model for FLOPs calculation...")
    # 创建一个全新的临时模型实例
    temp_model = core.LightTFTv2_1(config, strat, best_strat, best_pos).to(device)

    # WaveLightTFT 需要两个输入: x 和 w
    dummy_x = torch.randn(1, config.seq_len, config.input_dim).to(device)
    dummy_w = torch.randn(1, config.seq_len, config.input_dim).to(device)

    try:
        flops_str, params_str, _, _ = profile_fixed(temp_model, (dummy_x, dummy_w), fmt="%.3f")
        del temp_model # 立即释放
        torch.cuda.empty_cache()
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
        plt.title('WaveLightTFT Loss Curve')
        plt.legend()
        plt.savefig(os.path.join(plot_path, 'loss.png'))
        plt.close()
    
    cm = np.array(metrics['Confusion_Matrix'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title('WaveLightTFT Confusion Matrix')
    plt.savefig(os.path.join(plot_path, 'cm.png'))
    plt.close()
    
    print("🎨 Generating t-SNE plot...")
    if len(features) == 0:
        return

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
        plt.title('WaveLightTFT t-SNE Visualization')
        plt.legend()
        plt.savefig(os.path.join(plot_path, 'tsne.png'))
        plt.close()
        print("✅ t-SNE plot saved.")
    except Exception as e:
        print(f"❌ t-SNE Error: {e}")

# ================= 5. 主流程 =================
def run():
    print(f"🔥 启动 WaveLightTFT (Dynamic) 评估流程 (With JSON & FLOPs)")
    print(f"   输出目录: {Config.output_dir}")
    
    BEST_BASE, BEST_LEVEL, BEST_STRAT, BEST_POS = "db4", 2, "add", "middle"
    WEIGHT_STRAT = "dynamic_adaptive"
    
    # 0. 计算 FLOPs (在加载主模型前完成，避免冲突)
    flops_str, params_str = get_lightweight_metrics(Config, WEIGHT_STRAT, BEST_STRAT, BEST_POS, Config.device)
    print(f"💻 Model Stats -> FLOPs: {flops_str} | Params: {params_str}")

    # 1. 加载数据
    ds_tr, ds_val, ds_te = core.load_data(Config, BEST_BASE, BEST_LEVEL, DATA_CONFIG)
    ld_tr = DataLoader(ds_tr, Config.batch_size, shuffle=True, num_workers=0)
    ld_val = DataLoader(ds_val, Config.batch_size, num_workers=0)
    ld_te = DataLoader(ds_te, Config.batch_size, num_workers=0)
    
    # 2. 初始化主模型
    model = core.LightTFTv2_1(Config, WEIGHT_STRAT, BEST_STRAT, BEST_POS).to(Config.device)
    
    # 3. 训练或加载
    train_losses, val_losses = [], []
    if os.path.exists(Config.model_save_path):
        print(f"📂 加载现有模型: {Config.model_save_path}")
        model.load_state_dict(torch.load(Config.model_save_path))
    else:
        print(f"⚙️ 重新训练模型...")
        optimizer = optim.Adam(model.parameters(), lr=Config.lr)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        best_val_f1 = 0.0; patience_cnt = 0
        
        for ep in range(Config.epochs):
            model.train()
            e_loss = 0
            for x, w, y in tqdm(ld_tr, desc=f"Ep {ep+1}"):
                x, w, y = x.to(Config.device), w.to(Config.device), y.to(Config.device)
                optimizer.zero_grad()
                out = model(x, w) 
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                e_loss += loss.item()
            train_losses.append(e_loss / len(ld_tr))
            
            model.eval()
            v_preds, v_labels = [], []
            with torch.no_grad():
                for x, w, y in ld_val:
                    x, w, y = x.to(Config.device), w.to(Config.device), y.to(Config.device)
                    out = model(x, w)
                    v_preds.extend(torch.sigmoid(out).cpu().numpy())
                    v_labels.extend(y.cpu().numpy())
            
            f1 = f1_score(np.array(v_labels), (np.array(v_preds)>0.5).astype(int))
            print(f"   Ep {ep+1} | Val F1: {f1:.4f}")
            val_losses.append(0)
            
            if f1 > best_val_f1:
                best_val_f1 = f1
                torch.save(model.state_dict(), Config.model_save_path)
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= Config.patience: break
            scheduler.step(f1)
        model.load_state_dict(torch.load(Config.model_save_path))
    
    # 4. 注册 Hook 进行特征提取
    register_feature_hook(model)
    
    # 5. 测试
    print("🧪 开始最终测试...")
    model.eval()
    t_probs, t_labels = [], []
    extracted_features.clear()
    
    start_time = time.time()
    with torch.no_grad():
        for x, w, y in tqdm(ld_te, desc="Testing"):
            x, w = x.to(Config.device), w.to(Config.device)
            out = model(x, w)
            t_probs.extend(torch.sigmoid(out).cpu().numpy())
            t_labels.extend(y.numpy())
            
    inf_time_sec = (time.time() - start_time) / len(ds_te)
    inf_time_ms = inf_time_sec * 1000
    
    t_probs = np.array(t_probs).flatten()
    t_labels = np.array(t_labels).flatten()
    
    if len(extracted_features) > 0:
        t_features = torch.cat(extracted_features, dim=0).numpy()
    else:
        t_features = np.zeros((len(t_labels), Config.fc_hidden_dimension))

    # 6. 指标汇总与保存
    metrics = calculate_metrics(t_labels, t_probs)
    
    final_results = {
        **metrics,
        "FLOPs": flops_str,
        "Params": params_str,
        "Inference_Time_ms": inf_time_ms,
        "Inference_Time_sec": inf_time_sec
    }

    print("\n" + "="*50)
    print(f"{'WaveLightTFT Final Results':^50}")
    print("="*50)
    print(f"Accuracy:  {final_results['Accuracy']:.4f}")
    print(f"F1 Score:  {final_results['F1-Score']:.4f}")
    print(f"AUPRC:     {final_results['AUPRC']:.4f}")
    print("-" * 50)
    print(f"FLOPs:     {final_results['FLOPs']}")
    print(f"Params:    {final_results['Params']}")
    print(f"Inf Time:  {final_results['Inference_Time_ms']:.4f} ms/sample")
    print("="*50)
    
    # 保存 JSON
    with open(Config.json_save_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"📄 完整指标已保存至 JSON: {Config.json_save_path}")

    # 保存 NPZ
    np.savez(Config.npz_save_path, probs=t_probs, labels=t_labels, features=t_features, inf_time=inf_time_sec)
    print(f"💾 数据结果已保存至 NPZ: {Config.npz_save_path}")
    
    # 绘图
    plot_results(train_losses, val_losses, metrics, t_features, t_labels, Config.output_dir)

if __name__ == "__main__":
    run()