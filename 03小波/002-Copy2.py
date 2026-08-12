# -*- coding: utf-8 -*-
"""
EXP-3.2 分解层数选择实验
功能：在确定 db4 为最优基的前提下，对比不同分解层数的效果。
对比项：Level 2, 3, 4, 以及 [2,3] 混合层
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
from thop_mha_fix import profile_fixed  # thop MHA 计数修复（Bug fix）
import tft_core as core

# ================= 配置区域 =================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 基础参数
    input_dim = 150
    seq_len = 32
    batch_size = 64
    
    # Light-TFT v2.1 结构参数
    fc_hidden_dimension = 64
    attn_heads = 2
    grn_hidden_dim = 64
    decoder_hidden_dim = 32
    dropout = 0.2
    
    # 策略参数 (Exp 3.1 固定)
    dynamic_window_size = 8
    attention_hidden_dim = 32
    fixed_orig_weight = 0.5
    
    # 训练参数
    epochs = 50
    lr = 5e-4
    patience = 10
    
    # 路径
    wavelet_cache_dir = "wavelet_cache"
    save_dir = "exp3_2_models"
    result_path = "results/exp_3_2_level_full.json"

DATA_CONFIG = {
    "train": {"X_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_train_X_T32.npy", "y_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_train_y_T32.npy"},
    "val": {"X_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_val_X_T32.npy", "y_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_val_y_T32.npy"},
    "test": {"X_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_test_X_T32.npy", "y_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_test_y_T32.npy"}
}

# ================= 辅助函数 (复用) =================
def get_detailed_metrics(model, loader, device):
    model.eval()
    preds, labels = [], []
    start_time = time.time()
    with torch.no_grad():
        for x, w, y in loader:
            x, w = x.to(device), w.to(device)
            out = model(x, w)
            preds.extend(out.cpu().numpy())
            labels.extend(y.cpu().numpy())
    
    total_time = time.time() - start_time
    infer_ms = (total_time * 1000) / len(loader.dataset)
    
    metrics = core.calculate_metrics(np.array(labels), np.array(preds))
    
    # 补充指标
    from sklearn.metrics import average_precision_score, recall_score
    p_np = np.array(preds).flatten()
    l_np = np.array(labels).flatten()
    p_bin = (torch.sigmoid(torch.from_numpy(p_np)) > 0.5).numpy().astype(int)
    metrics['recall'] = recall_score(l_np, p_bin, average='binary')
    metrics['auprc'] = average_precision_score(l_np, p_np)
    metrics['time_ms'] = infer_ms
    return metrics

def get_lightweight_metrics(model, device):
    model.eval()
    dummy_x = torch.randn(1, Config.seq_len, Config.input_dim).to(device)
    dummy_w = torch.randn(1, Config.seq_len, Config.input_dim).to(device)
    try:
        flops, params, _, _ = profile_fixed(model, (dummy_x, dummy_w), fmt="%.4f")
    except:
        flops, params = "N/A", "N/A"
    return {"params": params, "flops": flops}

# ================= 主流程 =================
def run():
    core.create_dirs()
    os.makedirs(Config.save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 1. 锁定最佳小波基 (来自 Exp 3.1)
    BEST_BASE = "db4"
    
    # 2. 实验变量：分解层数
    # [2, 3] 表示混合层特征
    LEVELS = [2, 3, 4, [2, 3]] 
    
    # 固定配置
    FIXED = {"strat_f": "simple_concat", "strat_w": "dynamic_adaptive", "pos": "early"}
    
    all_results = []
    print(f"🔥 启动 EXP-3.2: 分解层数对比 (Base={BEST_BASE})")
    print(f"   对比项: {LEVELS}")
    
    for level in LEVELS:
        lvl_name = str(level)
        print(f"\n🚀 Running Level: {lvl_name} ...")
        
        # 加载数据 (core会自动处理 list 类型的 level)
        ds_tr, ds_val, ds_te = core.load_data(Config, BEST_BASE, level, DATA_CONFIG)
        ld_tr = DataLoader(ds_tr, Config.batch_size, shuffle=True)
        ld_val = DataLoader(ds_val, Config.batch_size)
        ld_te = DataLoader(ds_te, Config.batch_size)
        
        model = core.LightTFTv2_1(Config, FIXED['strat_w'], FIXED['strat_f'], FIXED['pos']).to(Config.device)
        optimizer = optim.Adam(model.parameters(), lr=Config.lr)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        
        best_val_f1 = 0.0
        patience_cnt = 0
        model_save_path = os.path.join(Config.save_dir, f"best_L{lvl_name}.pth")
        
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
        
        # Test
        model.load_state_dict(torch.load(model_save_path))
        test_metrics = get_detailed_metrics(model, ld_te, Config.device)
        lw_metrics = get_lightweight_metrics(model, Config.device)
        
        res = {
            "level": lvl_name,
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
        
        with open(Config.result_path, "w") as f:
            json.dump(all_results, f, indent=4)
            
        print(f"   ✅ Finished L{lvl_name}: F1={res['f1']}")

    # ================= 打印汇总表格 =================
    print("\n" + "="*110)
    print(f"{'EXP-3.2 分解层数选择实验结果汇总 (Base: db4)':^110}")
    print("="*110)
    headers = ["Level", "F1 Score", "Precision", "Recall", "AUPRC", "FNR", "Params", "FLOPs", "Time(ms)"]
    row_fmt = "| {:<12} | {:<8} | {:<9} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} |"
    
    print(row_fmt.format(*headers))
    print("-" * 110)
    
    for r in all_results:
        print(row_fmt.format(
            r['level'], r['f1'], r['precision'], r['recall'], r['auprc'], r['fnr'],
            r['params'], r['flops'], r['time_ms']
        ))
    print("="*110)
    print(f"结果已保存至: {Config.result_path}")

if __name__ == "__main__":
    run()