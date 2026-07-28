# -*- coding: utf-8 -*-
"""
EXP-3.1 小波基选择实验 (指标增强版)
功能：对比 db4, db8, sym4, coif3, bior3.3 五种小波基的效果
增强：输出包含 F1, Prec, Recall, FNR, Params, FLOPs, Time 等全套指标
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
from thop import profile, clever_format
import tft_core as core

# ================= 配置区域 =================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据与模型参数
    input_dim = 150
    seq_len = 32
    batch_size = 64
    fc_hidden_dimension = 64
    attn_heads = 2
    grn_hidden_dim = 64
    decoder_hidden_dim = 32
    dropout = 0.2
    
    # 策略参数
    dynamic_window_size = 8
    attention_hidden_dim = 32
    fixed_orig_weight = 0.5
    
    # 训练参数
    epochs = 50
    lr = 5e-4
    patience = 10
    
    # 路径
    wavelet_cache_dir = "wavelet_cache"
    save_dir = "exp3_1_models"
    result_path = "results/exp_3_1_base_full.json"

# 数据路径 (请确认路径无误)
DATA_CONFIG = {
    "train": {"X_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_train_X_T32.npy", "y_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_train_y_T32.npy"},
    "val": {"X_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_val_X_T32.npy", "y_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_val_y_T32.npy"},
    "test": {"X_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_test_X_T32.npy", "y_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_test_y_T32.npy"}
}

# ================= 辅助函数 =================
def get_detailed_metrics(model, loader, device):
    """计算完整的性能指标和推理速度"""
    model.eval()
    preds, labels = [], []
    start_time = time.time()
    
    with torch.no_grad():
        for x, w, y in loader:
            x, w = x.to(device), w.to(device)
            out = model(x, w)
            preds.extend(out.cpu().numpy())
            labels.extend(y.cpu().numpy()) # 确保转回CPU
            
    total_time = time.time() - start_time
    infer_ms = (total_time * 1000) / len(loader.dataset) # ms per sample
    
    # 计算性能指标
    metrics = core.calculate_metrics(np.array(labels), np.array(preds))
    
    # 补充 AUPRC (calculate_metrics 里可能没有)
    from sklearn.metrics import average_precision_score, recall_score
    p_np = np.array(preds).flatten()
    l_np = np.array(labels).flatten()
    p_bin = (torch.sigmoid(torch.from_numpy(p_np)) > 0.5).numpy().astype(int)
    
    metrics['recall'] = recall_score(l_np, p_bin, average='binary')
    metrics['auprc'] = average_precision_score(l_np, p_np)
    metrics['time_ms'] = infer_ms
    
    return metrics

def get_lightweight_metrics(model, device):
    """计算参数量和FLOPs"""
    model.eval()
    dummy_x = torch.randn(1, Config.seq_len, Config.input_dim).to(device)
    dummy_w = torch.randn(1, Config.seq_len, Config.input_dim).to(device)
    try:
        flops, params = profile(model, inputs=(dummy_x, dummy_w), verbose=False)
        flops, params = clever_format([flops, params], "%.4f")
    except Exception as e:
        print(f"⚠️ FLOPs calculation failed: {e}")
        flops, params = "N/A", "N/A"
    return {"params": params, "flops": flops}

# ================= 主流程 =================
def run():
    core.create_dirs()
    os.makedirs(Config.save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 实验变量：5种小波基
    BASES = ["db4", "db8", "sym4", "coif3", "bior3.3"]
    # 固定配置 (基于 LSTM+FC+Attention 架构)
    FIXED = {"level": 3, "strat_f": "simple_concat", "strat_w": "dynamic_adaptive", "pos": "early"}
    
    all_results = []
    print(f"🔥 启动 EXP-3.1: 小波基对比实验 (Count: {len(BASES)})")
    
    for base in BASES:
        print(f"\n🚀 Running Base: {base} ...")
        
        # 1. 加载数据
        ds_tr, ds_val, ds_te = core.load_data(Config, base, FIXED['level'], DATA_CONFIG)
        ld_tr = DataLoader(ds_tr, Config.batch_size, shuffle=True)
        ld_val = DataLoader(ds_val, Config.batch_size)
        ld_te = DataLoader(ds_te, Config.batch_size)
        
        # 2. 初始化模型
        model = core.LightTFTv2_1(Config, FIXED['strat_w'], FIXED['strat_f'], FIXED['pos']).to(Config.device)
        optimizer = optim.Adam(model.parameters(), lr=Config.lr)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        
        # 3. 训练循环
        best_val_f1 = 0.0
        patience_cnt = 0
        model_save_path = os.path.join(Config.save_dir, f"best_{base}.pth")
        
        for ep in range(Config.epochs):
            model.train()
            train_loss = 0
            for x, w, y in ld_tr:
                x, w, y = x.to(Config.device), w.to(Config.device), y.to(Config.device)
                optimizer.zero_grad()
                out = model(x, w)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 验证
            val_metrics = get_detailed_metrics(model, ld_val, Config.device)
            f1 = val_metrics['f1']
            
            if ep % 5 == 0:
                print(f"   Ep {ep+1}: Loss {train_loss/len(ld_tr):.4f} | Val F1 {f1:.4f}")
            
            scheduler.step(f1)
            
            if f1 > best_val_f1:
                best_val_f1 = f1
                patience_cnt = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                patience_cnt += 1
                if patience_cnt >= Config.patience:
                    print(f"   ⏹ Early stop at Ep {ep+1}, Best Val F1: {best_val_f1:.4f}")
                    break
        
        # 4. 最终测试 (加载最佳模型)
        model.load_state_dict(torch.load(model_save_path))
        test_metrics = get_detailed_metrics(model, ld_te, Config.device)
        lw_metrics = get_lightweight_metrics(model, Config.device)
        
        # 整合结果
        res = {
            "base": base,
            "f1": round(test_metrics['f1'], 4),
            "precision": round(test_metrics['prec'], 4),
            "recall": round(test_metrics['recall'], 4),
            "auprc": round(test_metrics['auprc'], 4),
            "fnr": round(test_metrics['fnr'], 4),
            "params": lw_metrics['params'],
            "flops": lw_metrics['flops'],
            "time_ms": round(test_metrics['time_ms'], 3)
        }
        all_results.append(res)
        
        # 实时保存
        with open(Config.result_path, "w") as f:
            json.dump(all_results, f, indent=4)
            
        print(f"   ✅ Finished {base}: F1={res['f1']}, Time={res['time_ms']}ms")

    # ================= 打印汇总表格 =================
    print("\n" + "="*110)
    print(f"{'EXP-3.1 小波基选择实验结果汇总':^110}")
    print("="*110)
    headers = ["Wavelet Base", "F1 Score", "Precision", "Recall", "AUPRC", "FNR", "Params", "FLOPs", "Time(ms)"]
    row_fmt = "| {:<12} | {:<8} | {:<9} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} |"
    
    print(row_fmt.format(*headers))
    print("-" * 110)
    
    for r in all_results:
        print(row_fmt.format(
            r['base'], r['f1'], r['precision'], r['recall'], r['auprc'], r['fnr'],
            r['params'], r['flops'], r['time_ms']
        ))
    print("="*110)
    print(f"结果已保存至: {Config.result_path}")

if __name__ == "__main__":
    run()