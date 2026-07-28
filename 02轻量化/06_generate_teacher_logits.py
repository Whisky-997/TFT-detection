# -*- coding: utf-8 -*-
"""
文件名: generate_teacher_logits.py
作用: 离线计算 Teacher 模型的 Logits 并保存，为极速蒸馏做准备。
注意: 此脚本只需运行一次。
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# 尝试导入 Teacher 定义
try:
    from tft_binary import TFTBinary
except ImportError:
    raise ImportError("❌ 错误: 当前目录下缺少 'tft_binary.py'，无法加载 Teacher 模型。")

# ================= 配置 =================
class GenConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 原始数据路径 (与你提供的一致)
    DATA_ROOT = "/root/autodl-tmp/graduate-thesis/data/tensor_T32"
    TRAIN_X = os.path.join(DATA_ROOT, "cic17_W32_S16_train_X_T32.npy")
    VAL_X   = os.path.join(DATA_ROOT, "cic17_W32_S16_val_X_T32.npy")
    # Test集通常不需要蒸馏，只需最后评估，所以这里不生成Test的Logits
    
    # Teacher 权重路径
    TEACHER_WEIGHTS = "/root/autodl-tmp/graduate-thesis/duibi/tft_medium_output/tft_baseline_best.pth"
    
    # 输出路径 (保存生成的 logits)
    OUTPUT_DIR = "/root/autodl-tmp/graduate-thesis/data/distill_logits"
    
    # Teacher 参数 (严格匹配 TFT-Medium)
    TEACHER_PARAMS = {
        "time_varying_real_variables_encoder": 150,
        "time_varying_real_variables_decoder": 150,
        "seq_length": 32,
        "lstm_hidden_dimension": 64,
        "lstm_layers": 2,
        "attn_heads": 4,
        "dropout": 0.3,
        "embedding_dim": 64,
        "batch_size": 128, # 推理时可以用大 Batch 加速
        "return_sequence": False,
        "device": device
    }

if not os.path.exists(GenConfig.OUTPUT_DIR):
    os.makedirs(GenConfig.OUTPUT_DIR)

def get_logits(model, x_path, desc):
    """加载数据，运行模型，返回 logits"""
    print(f"📥 加载数据: {x_path}")
    data = np.load(x_path).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(data))
    # num_workers=4 加速数据读取
    loader = DataLoader(dataset, batch_size=GenConfig.TEACHER_PARAMS['batch_size'], shuffle=False, num_workers=4)
    
    all_logits = []
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            x = batch[0].to(GenConfig.device)
            # Teacher forward
            out = model(x)
            # 处理可能的 tuple 输出 (TFTBinary 有时返回 (logits, attention))
            logits = out[0] if isinstance(out, tuple) else out
            all_logits.append(logits.cpu().numpy())
            
    return np.concatenate(all_logits, axis=0)

def main():
    print("🚀 开始生成 Teacher Logits (离线蒸馏准备)...")
    
    # 1. 初始化 Teacher
    print("👨‍🏫 加载 Teacher 模型...")
    teacher = TFTBinary(GenConfig.TEACHER_PARAMS).to(GenConfig.device)
    
    if os.path.exists(GenConfig.TEACHER_WEIGHTS):
        teacher.load_state_dict(torch.load(GenConfig.TEACHER_WEIGHTS))
        print("✅ Teacher 权重已加载")
    else:
        raise FileNotFoundError(f"❌ 找不到 Teacher 权重: {GenConfig.TEACHER_WEIGHTS}")
    
    # 2. 处理训练集
    train_logits = get_logits(teacher, GenConfig.TRAIN_X, "Processing Train")
    save_path_train = os.path.join(GenConfig.OUTPUT_DIR, "teacher_logits_train.npy")
    np.save(save_path_train, train_logits)
    print(f"💾 Train Logits 已保存: {save_path_train} | Shape: {train_logits.shape}")
    
    # 3. 处理验证集
    val_logits = get_logits(teacher, GenConfig.VAL_X, "Processing Val")
    save_path_val = os.path.join(GenConfig.OUTPUT_DIR, "teacher_logits_val.npy")
    np.save(save_path_val, val_logits)
    print(f"💾 Val Logits 已保存: {save_path_val} | Shape: {val_logits.shape}")
    
    print("\n🎉 全部完成！现在可以运行 distill_offline_fast.py 进行极速蒸馏了。")

if __name__ == "__main__":
    main()