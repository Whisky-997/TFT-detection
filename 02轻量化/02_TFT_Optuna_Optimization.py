# -*- coding: utf-8 -*-
"""
02_TFT_Optuna_Robust_Sleep.py
【睡觉专用版】功能：
1. 极度鲁棒：遇到报错自动跳过当前 Trial，绝不卡死。
2. 详细日志：每个 Epoch 都输出 F1/Prec/Rec 等指标。
3. 断点续传：使用 SQLite 数据库，随时可恢复。
"""

import os
import sys
import time
import logging
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, confusion_matrix

# 引入 Optuna
try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    print("❌ 严重错误：未安装 optuna。请运行 pip install optuna")
    sys.exit(1)

# 引入模型
try:
    from tft_binary import TFTBinary
except ImportError:
    print("❌ 严重错误：未找到 tft_binary.py。")
    sys.exit(1)

# ================= 全局配置 =================
BASE_DIR = "/root/autodl-tmp/graduate-thesis/duibi"
OUTPUT_DIR = os.path.join(BASE_DIR, "tft_optuna_sleep_run")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据库路径 (核心：用于断点续传)
DB_URL = f"sqlite:///{os.path.join(OUTPUT_DIR, 'optuna_study.db')}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS = 10       # 总共跑10轮
MAX_EPOCHS_SEARCH = 10  # 搜索阶段 Epoch
MAX_EPOCHS_FINAL = 30   # 最终训练 Epoch

# ================= 日志系统 (实时刷写) =================
log_file = os.path.join(OUTPUT_DIR, "tft_running_detail.log")

# 自定义 Logger，确保立即写入文件
logger = logging.getLogger("TFT-Sleep")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 文件句柄
fh = logging.FileHandler(log_file, encoding='utf-8')
fh.setFormatter(formatter)
logger.addHandler(fh)

# 控制台句柄
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)

# 强制 Optuna 日志也输出到这里
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()

# ================= 数据集 =================
DATA_CONFIG = {
    "train": {"X": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_train_X_T32.npy", 
              "y": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_train_y_T32.npy"},
    "val":   {"X": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_val_X_T32.npy", 
              "y": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_val_y_T32.npy"},
    "test":  {"X": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_test_X_T32.npy", 
              "y": "/root/autodl-tmp/graduate-thesis/data/tensor_T32/cic17_W32_S16_test_y_T32.npy"}
}

class SimpleDataset(Dataset):
    def __init__(self, X_path, y_path):
        try:
            self.X = np.load(X_path).astype(np.float32)
            self.y = np.load(y_path).astype(np.float32)
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {X_path} - {e}")
            raise e
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): 
        return torch.tensor(self.X[idx], dtype=torch.float), torch.tensor(self.y[idx], dtype=torch.float)

# ================= 辅助函数 =================
def calculate_metrics(labels, preds_prob):
    try:
        preds_bin = np.argmax(preds_prob, axis=1)
        cm = confusion_matrix(labels, preds_bin)
        if cm.shape == (2, 2): tn, fp, fn, tp = cm.ravel()
        else: fn = 0; tp = 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return {
            "F1": f1_score(labels, preds_bin),
            "Prec": precision_score(labels, preds_bin, zero_division=0),
            "Rec": recall_score(labels, preds_bin, zero_division=0),
            "FNR": fnr
        }
    except Exception:
        return {"F1": 0, "Prec": 0, "Rec": 0, "FNR": 0}

# ================= 1. 搜索目标函数 (防弹版) =================
def objective(trial):
    trial_idx = trial.number + 1
    logger.info(f"👉 [开始 Trial {trial_idx}/{N_TRIALS}]")
    
    try:
        # 1. 参数采样
        params = {
            "time_varying_real_variables_encoder": 150,
            "time_varying_real_variables_decoder": 150,
            "seq_length": 32,
            "return_sequence": False,
            "device": DEVICE,
            "attn_heads": 4, 
            # 搜索空间
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "lstm_hidden_dimension": trial.suggest_categorical("lstm_hidden_dimension", [64, 128]),
            "lstm_layers": trial.suggest_int("lstm_layers", 1, 2)
        }
        
        # 2. 数据加载
        ds_tr = SimpleDataset(DATA_CONFIG["train"]["X"], DATA_CONFIG["train"]["y"])
        ds_va = SimpleDataset(DATA_CONFIG["val"]["X"], DATA_CONFIG["val"]["y"])
        ld_tr = DataLoader(ds_tr, params["batch_size"], shuffle=True, num_workers=0)
        ld_va = DataLoader(ds_va, params["batch_size"], num_workers=0)
        
        # 3. 模型初始化
        model = TFTBinary(params).to(DEVICE)
        
        # 4. 训练准备
        y_tr = ds_tr.y.flatten()
        pos_w = (len(y_tr) - y_tr.sum()) / (y_tr.sum() + 1e-5)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w], dtype=torch.float).to(DEVICE))
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        
        # 5. 训练循环
        best_val_loss = float('inf')
        
        for epoch in range(MAX_EPOCHS_SEARCH):
            # --- Train ---
            model.train()
            tr_loss = 0
            for x, y in ld_tr:
                x, y = x.to(DEVICE), y.to(DEVICE).long()
                optimizer.zero_grad()
                out = model(x); logits = out[0] if isinstance(out, tuple) else out
                loss = criterion(logits, y)
                loss.backward(); optimizer.step()
                tr_loss += loss.item()
            avg_tr_loss = tr_loss / len(ld_tr)
            
            # --- Val ---
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
            met = calculate_metrics(np.array(targs), np.array(preds))
            
            # --- 详细日志 ---
            logger.info(f"   [T{trial_idx}] Ep {epoch+1:02d} | Loss: {avg_tr_loss:.4f} / {avg_va_loss:.4f} | "
                        f"F1: {met['F1']:.4f} | Prec: {met['Prec']:.4f} | Rec: {met['Rec']:.4f}")
            
            # --- Optuna 报告 ---
            trial.report(avg_va_loss, epoch)
            if trial.should_prune():
                logger.info(f"✂️  Trial {trial_idx} 效果不佳，已剪枝 (Pruned)")
                raise optuna.exceptions.TrialPruned()
            
            best_val_loss = min(best_val_loss, avg_va_loss)

        return best_val_loss

    except optuna.exceptions.TrialPruned as e:
        raise e # 正常剪枝，抛出给 Optuna 处理
    except Exception as e:
        logger.error(f"⚠️  Trial {trial_idx} 发生异常，跳过此轮。错误: {str(e)}")
        traceback.print_exc()
        # 返回一个很大的 Loss，让 Optuna 知道这轮失败了
        return float('inf')

# ================= 2. 最终训练 (防弹版) =================
def train_best_model(best_params):
    logger.info("="*60)
    logger.info("🏆 搜索完成，开始训练最佳模型 (Final Training)")
    
    try:
        final_params = {
            "time_varying_real_variables_encoder": 150,
            "time_varying_real_variables_decoder": 150,
            "seq_length": 32,
            "return_sequence": False,
            "device": DEVICE,
            "attn_heads": 4,
            **best_params
        }
        
        ds_tr = SimpleDataset(DATA_CONFIG["train"]["X"], DATA_CONFIG["train"]["y"])
        ds_va = SimpleDataset(DATA_CONFIG["val"]["X"], DATA_CONFIG["val"]["y"])
        ds_te = SimpleDataset(DATA_CONFIG["test"]["X"], DATA_CONFIG["test"]["y"])
        
        ld_tr = DataLoader(ds_tr, final_params["batch_size"], shuffle=True, num_workers=0)
        ld_va = DataLoader(ds_va, final_params["batch_size"], num_workers=0)
        ld_te = DataLoader(ds_te, final_params["batch_size"], num_workers=0)
        
        model = TFTBinary(final_params).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0], dtype=torch.float).to(DEVICE))
        optimizer = optim.Adam(model.parameters(), lr=final_params["lr"])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        best_loss = float('inf')
        best_path = os.path.join(OUTPUT_DIR, "tft_best_final.pth")
        
        for ep in range(MAX_EPOCHS_FINAL):
            model.train()
            for x, y in ld_tr:
                x, y = x.to(DEVICE), y.to(DEVICE).long()
                optimizer.zero_grad()
                loss = criterion(model(x)[0], y)
                loss.backward(); optimizer.step()
            
            model.eval()
            va_loss = 0
            preds, targs = [], []
            with torch.no_grad():
                for x, y in ld_va:
                    x, y = x.to(DEVICE), y.to(DEVICE).long()
                    logits = model(x)[0]
                    va_loss += criterion(logits, y).item()
                    probs = torch.softmax(logits, dim=1)
                    preds.extend(probs.cpu().numpy()); targs.extend(y.cpu().numpy())
                    
            avg_va_loss = va_loss / len(ld_va)
            met = calculate_metrics(np.array(targs), np.array(preds))
            
            logger.info(f"🏅 Final Ep {ep+1:02d} | Val Loss: {avg_va_loss:.4f} | F1: {met['F1']:.4f}")
            scheduler.step(avg_va_loss)
            
            # 安全保存
            if avg_va_loss < best_loss:
                best_loss = avg_va_loss
                try:
                    torch.save(model.state_dict(), best_path)
                    logger.info("   💾 模型已保存")
                except Exception as e:
                    logger.warning(f"   ⚠️ 模型保存失败 (不影响训练): {e}")

    except Exception as e:
        logger.error("❌ 最终训练阶段发生严重错误，但前面的搜索结果已保存。")
        traceback.print_exc()

# ================= 主程序 =================
if __name__ == "__main__":
    logger.info(f"🚀 启动 'Sleep Mode' HPO (DB: {DB_URL})")
    
    # 1. 创建数据库存储 (支持断点续传)
    storage = optuna.storages.RDBStorage(url=DB_URL)
    study = optuna.create_study(study_name="tft_sleep_study", storage=storage, load_if_exists=True, direction="minimize")
    
    remaining = N_TRIALS - len(study.trials)
    if remaining > 0:
        logger.info(f"🔍 计划执行 {remaining} 轮搜索...")
        # 【关键】catch=(Exception,) 告诉 Optuna 捕获所有常规异常，不要崩溃
        study.optimize(objective, n_trials=remaining, catch=(Exception,))
    else:
        logger.info("✅ 所有轮次已完成。")

    # 2. 备份数据 (即使报错也不停止)
    try:
        df = study.trials_dataframe()
        csv_path = os.path.join(OUTPUT_DIR, "optuna_history_final.csv")
        df.to_csv(csv_path)
        logger.info(f"📝 历史数据已备份至 {csv_path}")
    except Exception as e:
        logger.warning(f"⚠️ CSV 备份失败: {e}")

    # 3. 跑最好的模型
    if len(study.trials) > 0:
        logger.info(f"🏆 最佳 Loss: {study.best_trial.value:.4f}")
        train_best_model(study.best_trial.params)
    
    logger.info("💤 任务全部结束，晚安！")