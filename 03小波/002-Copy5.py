# -*- coding: utf-8 -*-
"""
EXP-3.5 加权策略选择实验 (Final Step)
功能：在最强配置下 (db4 + L2 + Add + Middle)，对比 'Fixed' vs 'Dynamic' 加权。
目的：验证动态门控机制是否比固定权重更有效。
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

# ================= 动态注入 Add 策略 (必须保留) =================
class AddModule(nn.Module):
    def __init__(self, cfg): super().__init__()
    def forward(self, x, w): return x + w

def patched_get_fusion_mod(name, cfg):
    if name == "simple_concat": return core.SimpleConcatModule(cfg)
    if name == "light_attention": return core.LightAttentionModule(cfg)
    if name == "add": return AddModule(cfg) # 必须要有这个
    raise ValueError(f"Unknown fusion: {name}")

core.get_fusion_mod = patched_get_fusion_mod
print("✅ 已动态注入 Add 融合策略支持")

# ================= 配置区域 =================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 150; seq_len = 32; batch_size = 64
    fc_hidden_dimension = 64; attn_heads = 2; grn_hidden_dim = 64; decoder_hidden_dim = 32; dropout = 0.2
    
    # 策略参数
    dynamic_window_size = 8; attention_hidden_dim = 32; fixed_orig_weight = 0.5
    
    # 训练参数
    epochs = 50; lr = 5e-4; patience = 10
    
    # 路径
    wavelet_cache_dir = "wavelet_cache"
    save_dir = "exp3_5_models"
    result_path = "results/exp_3_5_weight_full.json"

DATA_CONFIG = {
    "train": {"X_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_train_X_T32.npy", "y_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_train_y_T32.npy"},
    "val": {"X_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_val_X_T32.npy", "y_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_val_y_T32.npy"},
    "test": {"X_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_test_X_T32.npy", "y_path": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_test_y_T32.npy"}
}

# ================= 辅助函数 =================
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
        flops, params = profile(model, inputs=(dummy_x, dummy_w), verbose=False)
        flops, params = clever_format([flops, params], "%.4f")
    except:
        flops, params = "N/A", "N/A"
    return {"params": params, "flops": flops}

# ================= 主流程 =================
def run():
    core.create_dirs()
    os.makedirs(Config.save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 👑 众神归位：集结之前的所有最优解
    BEST_BASE = "db4"
    BEST_LEVEL = 2
    BEST_STRAT = "add"
    BEST_POS = "middle"  # 这里改成了 middle！
    
    # 实验变量：加权策略
    # dynamic_adaptive: 可学习的窗口权重
    # fixed_weight: 固定 0.5:0.5 (基准)
    WEIGHTS = ["dynamic_adaptive", "fixed_weight"]
    
    print(f"🔥 启动 EXP-3.5: 加权策略对比")
    print(f"   配置: Base={BEST_BASE}, Level={BEST_LEVEL}, Strat={BEST_STRAT}, Pos={BEST_POS}")
    
    ds_tr, ds_val, ds_te = core.load_data(Config, BEST_BASE, BEST_LEVEL, DATA_CONFIG)
    ld_tr = DataLoader(ds_tr, Config.batch_size, shuffle=True)
    ld_val = DataLoader(ds_val, Config.batch_size)
    ld_te = DataLoader(ds_te, Config.batch_size)
    
    all_results = []
    
    for w_strat in WEIGHTS:
        print(f"\n🚀 Running Weight: {w_strat} ...")
        
        model = core.LightTFTv2_1(Config, w_strat, BEST_STRAT, BEST_POS).to(Config.device)
        optimizer = optim.Adam(model.parameters(), lr=Config.lr)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        
        best_val_f1 = 0.0
        patience_cnt = 0
        model_save_path = os.path.join(Config.save_dir, f"best_{w_strat}.pth")
        
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
            "weight": w_strat,
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
            
        print(f"   ✅ Finished {w_strat}: F1={res['f1']}")

    # ================= 打印汇总表格 =================
    print("\n" + "="*110)
    print(f"{'EXP-3.5 加权策略选择实验结果汇总':^110}")
    print("="*110)
    headers = ["Weight", "F1 Score", "Precision", "Recall", "AUPRC", "FNR", "Params", "FLOPs", "Time(ms)"]
    row_fmt = "| {:<16} | {:<8} | {:<9} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} |"
    
    print(row_fmt.format(*headers))
    print("-" * 110)
    
    for r in all_results:
        print(row_fmt.format(
            r['weight'], r['f1'], r['precision'], r['recall'], r['auprc'], r['fnr'],
            r['params'], r['flops'], r['time_ms']
        ))
    print("="*110)
    print(f"结果已保存至: {Config.result_path}")

if __name__ == "__main__":
    run()