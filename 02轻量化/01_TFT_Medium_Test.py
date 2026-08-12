# -*- coding: utf-8 -*-
"""
01_TFT_Medium_Final_Fix.py
修复点：
1. 强制 CrossEntropyLoss 的 weight 参数为 torch.float (Float32)，解决 Double 类型不匹配报错。
2. 确保 Dataset 返回的数据显式转换为 float32。
"""
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, confusion_matrix

# 引入模型定义
try:
    from tft_binary import TFTBinary
except ImportError:
    raise ImportError("请确保 'tft_binary.py' 文件在当前目录下！")

# ================= 配置 =================
BASE_DIR = "/root/autodl-tmp/graduate-thesis/duibi"
OUTPUT_DIR = os.path.join(BASE_DIR, "tft_medium_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 关键文件路径
MODEL_PATH = os.path.join(OUTPUT_DIR, "tft_baseline_best.pth")
LOG_PATH = os.path.join(OUTPUT_DIR, "tft_medium.log")
RESULT_PATH = os.path.join(OUTPUT_DIR, "tft_medium_results.npz")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_CONFIG = {
    "train": {"X": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_train_X_T32.npy", 
              "y": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_train_y_T32.npy"},
    "val":   {"X": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_val_X_T32.npy", 
              "y": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_val_y_T32.npy"},
    "test":  {"X": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_test_X_T32.npy", 
              "y": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_test_y_T32.npy"}
}

# ================= 工具函数 =================
def setup_logger():
    logger = logging.getLogger("TFT-Medium")
    logger.setLevel(logging.INFO)
    if logger.handlers: logger.handlers.clear()
    
    fh = logging.FileHandler(LOG_PATH, encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter); ch.setFormatter(formatter)
    
    logger.addHandler(fh); logger.addHandler(ch)
    return logger

class SimpleDataset(Dataset):
    def __init__(self, X_path, y_path):
        # 强制转换为 float32
        self.X = np.load(X_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): 
        # 返回时再次确保 dtype
        return torch.tensor(self.X[idx], dtype=torch.float), torch.tensor(self.y[idx], dtype=torch.float)

def calculate_metrics(labels, preds_prob):
    preds_bin = np.argmax(preds_prob, axis=1)
    pos_probs = preds_prob[:, 1]
    
    cm = confusion_matrix(labels, preds_bin)
    if cm.shape == (2, 2): tn, fp, fn, tp = cm.ravel()
    else: fn = 0; tp = 0; tn = 0; fp = 0 # Handle edge cases
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        "F1": f1_score(labels, preds_bin),
        "Prec": precision_score(labels, preds_bin, zero_division=0),
        "Rec": recall_score(labels, preds_bin, zero_division=0),
        "AUPRC": average_precision_score(labels, pos_probs),
        "FNR": fnr,
        "CM": cm
    }

# ================= 主程序 =================
def main():
    logger = setup_logger()
    logger.info("🚀 启动 TFT Medium (Baseline) 全面训练 [修复版]")
    logger.info(f"📁 输出目录: {OUTPUT_DIR}")
    
    # 1. 参数配置
    PARAMS = {
        "time_varying_real_variables_encoder": 150,
        "time_varying_real_variables_decoder": 150,
        "seq_length": 32,
        "lstm_hidden_dimension": 64,   # 64维基准
        "lstm_layers": 2,
        "attn_heads": 4,
        "dropout": 0.3,                
        "embedding_dim": 64,           
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 30,
        "patience": 10,
        "return_sequence": False,
        "device": DEVICE
    }
    logger.info(f"⚙️ 参数配置: {PARAMS}")
    
    # 2. 数据加载
    logger.info("📥 加载数据...")
    ds_tr = SimpleDataset(DATA_CONFIG["train"]["X"], DATA_CONFIG["train"]["y"])
    ds_va = SimpleDataset(DATA_CONFIG["val"]["X"], DATA_CONFIG["val"]["y"])
    ds_te = SimpleDataset(DATA_CONFIG["test"]["X"], DATA_CONFIG["test"]["y"])
    
    ld_tr = DataLoader(ds_tr, PARAMS["batch_size"], shuffle=True, num_workers=0)
    ld_va = DataLoader(ds_va, PARAMS["batch_size"], num_workers=0)
    ld_te = DataLoader(ds_te, PARAMS["batch_size"], num_workers=0)
    
    # 3. 模型初始化
    model = TFTBinary(PARAMS).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"📈 模型参数量: {total_params:,} (Baseline Reference)")
    
    # 4. 训练组件 (关键修复处)
    y_tr = ds_tr.y.flatten()
    pos_w = (len(y_tr) - y_tr.sum()) / (y_tr.sum() + 1e-5)
    
    # 【修复】强制 dtype=torch.float，防止生成 Double Tensor
    weight_tensor = torch.tensor([1.0, pos_w], dtype=torch.float).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=PARAMS["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 5. 训练循环
    best_loss = float('inf')
    patience_cnt = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    logger.info("🔄 开始训练...")
    start_train_time = time.time()
    
    for ep in range(PARAMS["epochs"]):
        model.train()
        tr_loss = 0
        for x, y in ld_tr:
            # y 需要是 long 类型用于 CrossEntropy，但数据本身是 float32 加载的，这里转 long
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            optimizer.zero_grad()
            out = model(x); logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            tr_loss += loss.item()
            
        avg_tr_loss = tr_loss / len(ld_tr)
        
        # Validation
        model.eval()
        va_loss = 0
        preds, targs = [], []
        with torch.no_grad():
            for x, y in ld_va:
                x, y = x.to(DEVICE), y.to(DEVICE).long()
                out = model(x); logits = out[0] if isinstance(out, tuple) else out
                va_loss += criterion(logits, y).item()
                probs = torch.softmax(logits, dim=1)
                preds.extend(probs.cpu().numpy()); targs.extend(y.cpu().numpy())
        
        avg_va_loss = va_loss / len(ld_va)
        metrics = calculate_metrics(np.array(targs), np.array(preds))
        
        # 记录
        history['train_loss'].append(avg_tr_loss)
        history['val_loss'].append(avg_va_loss)
        history['val_f1'].append(metrics['F1'])
        
        logger.info(f"Ep {ep+1:02d} | Loss: {avg_tr_loss:.4f} | Val Loss: {avg_va_loss:.4f} | "
                    f"F1: {metrics['F1']:.4f} | Rec: {metrics['Rec']:.4f}")
        
        scheduler.step(avg_va_loss)
        
        # 早停逻辑
        if avg_va_loss < best_loss:
            best_loss = avg_va_loss
            torch.save(model.state_dict(), MODEL_PATH)
            logger.info(f"   💾 New Best Model Saved (Loss: {best_loss:.4f})")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PARAMS["patience"]:
                logger.info("🛑 Early Stopping Triggered")
                break
    
    total_train_time = time.time() - start_train_time
    logger.info(f"⏱️ 总训练时间: {total_train_time/60:.1f} min")

    # 6. 最终测试
    logger.info("\n🧪 正在评估最佳模型 (Test Set)...")
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        logger.warning("未找到保存的模型权重，使用当前模型进行测试")
        
    model.eval()
    
    test_preds, test_targs = [], []
    t_start = time.time()
    
    with torch.no_grad():
        for x, y in ld_te:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            out = model(x); logits = out[0] if isinstance(out, tuple) else out
            probs = torch.softmax(logits, dim=1)
            test_preds.extend(probs.cpu().numpy()); test_targs.extend(y.cpu().numpy())
            
    infer_time = (time.time() - t_start) / len(ds_te) * 1000 # ms/sample
    
    final_metrics = calculate_metrics(np.array(test_targs), np.array(test_preds))
    
    logger.info("="*60)
    logger.info("📊 Baseline (Medium TFT) Final Results")
    logger.info("-" * 30)
    logger.info(f"F1-Score  : {final_metrics['F1']:.4f}")
    logger.info(f"Precision : {final_metrics['Prec']:.4f}")
    logger.info(f"Recall    : {final_metrics['Rec']:.4f}")
    logger.info(f"AUPRC     : {final_metrics['AUPRC']:.4f}")
    logger.info(f"FNR       : {final_metrics['FNR']:.4f}")
    logger.info(f"Time      : {infer_time:.4f} ms/sample")
    logger.info(f"Params    : {total_params:,}")
    logger.info("-" * 30)
    
    np.savez(RESULT_PATH, 
             train_loss=history['train_loss'],
             val_loss=history['val_loss'],
             test_preds=test_preds,
             test_labels=test_targs,
             metrics=final_metrics)
    logger.info(f"📝 结果已保存至: {RESULT_PATH}")

if __name__ == "__main__":
    main()