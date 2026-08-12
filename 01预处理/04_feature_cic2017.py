import pandas as pd
import numpy as np
import os
import pywt
import joblib
from sklearn.preprocessing import MinMaxScaler

def calculate_wavelet_energy(data):
    if len(data) < 2: return 0.0
    try:
        cA, cD = pywt.dwt(data, 'haar')
        energy = np.sum(cD ** 2)
        return np.log1p(energy) # log1p 平滑
    except: return 0.0

def build_features():
    base_dir = "/root/autodl-tmp/graduate-thesis/data/cleaned"
    output_dir = "/root/autodl-tmp/graduate-thesis/data/feature"
    scaler_path = "/root/autodl-tmp/graduate-thesis/data/feature/scalers"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(scaler_path, exist_ok=True)
    
    # 配置
    WINDOW_SIZE = 32
    SLIDE_STEP = 16
    ABNORMAL_RATIO = 0.15
    
    # 【核心修正】恢复完整的 30 个基础特征列表 (30 * 5 = 150维)
    CORE_FEATURES = [
        "Flow Duration", 
        "Total Fwd Packets", "Total Backward Packets", 
        "Total Length of Fwd Packets", "Total Length of Bwd Packets", 
        "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
        "Flow Bytes/s", "Flow Packets/s", 
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", 
        "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", 
        "Fwd PSH Flags", "Bwd PSH Flags", 
        "Fwd URG Flags", "Bwd URG Flags", 
        "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count"
    ]

    # 1. 分别加载数据
    print("✅ 加载数据集...")
    datasets = {}
    for tag in ["train", "val", "test"]:
        path = os.path.join(base_dir, f"cic2017_cleaned_{tag}.csv")
        if not os.path.exists(path):
            print(f"❌ 找不到 {path}"); return
        datasets[tag] = pd.read_csv(path)
        
    # 确认存在的特征列 (防止清洗时某些列被意外删除了)
    avail_feats = [c for c in CORE_FEATURES if c in datasets['train'].columns]
    print(f"✅ 计划提取特征: {len(CORE_FEATURES)} -> 实际有效特征: {len(avail_feats)}")
    
    if len(avail_feats) < 30:
        print("⚠️ 警告：部分特征在清洗后的数据中未找到，最终维度可能不足 150。")
        print(f"缺失特征: {set(CORE_FEATURES) - set(avail_feats)}")
    
    # 2. 原始数据归一化 (防泄露：Fit Train, Apply All)
    raw_scaler = MinMaxScaler((0, 1))
    
    # 处理 Inf/NaN
    for tag in datasets:
        datasets[tag] = datasets[tag].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    datasets['train'][avail_feats] = raw_scaler.fit_transform(datasets['train'][avail_feats])
    datasets['val'][avail_feats] = raw_scaler.transform(datasets['val'][avail_feats])
    datasets['test'][avail_feats] = raw_scaler.transform(datasets['test'][avail_feats])
    
    joblib.dump(raw_scaler, os.path.join(scaler_path, "raw_data_scaler.pkl"))
    print("✅ 原始数据归一化完成")
    
    # 3. 独立滑窗
    for tag in ["train", "val", "test"]:
        print(f"\n🚀 构建窗口特征: {tag} (W{WINDOW_SIZE}_S{SLIDE_STEP})")
        df_curr = datasets[tag].reset_index(drop=True)
        temporal_results = []
        
        total_windows = int(np.ceil((len(df_curr) - WINDOW_SIZE) / SLIDE_STEP)) + 1
        
        for w_idx in range(total_windows):
            start = w_idx * SLIDE_STEP
            end = min(start + WINDOW_SIZE, len(df_curr))
            
            if end - start < WINDOW_SIZE // 2: continue
            
            window_data = df_curr.iloc[start:end]
            stats = {"aux_window_index": w_idx}
            
            # 计算5种统计量 (30 * 5 = 150)
            for feat in avail_feats:
                vals = window_data[feat].values
                stats[f"{feat}_mean"] = np.mean(vals)
                stats[f"{feat}_std"] = np.std(vals)
                stats[f"{feat}_max"] = np.max(vals)
                stats[f"{feat}_min"] = np.min(vals)
                stats[f"{feat}_energy"] = calculate_wavelet_energy(vals)
            
            # 标签
            if len(window_data) > 0:
                is_abnormal = (window_data['Label'] == 1).sum() / len(window_data) > ABNORMAL_RATIO
                stats["window_label"] = 1 if is_abnormal else 0
            else:
                stats["window_label"] = 0
                
            temporal_results.append(stats)
            if w_idx % 2000 == 0: print(f"\r   进度: {w_idx}/{total_windows}", end="")
            
        res_df = pd.DataFrame(temporal_results)
        save_path = os.path.join(output_dir, f"cic2017_temporal_feature_W{WINDOW_SIZE}_S{SLIDE_STEP}_{tag}.csv")
        res_df.to_csv(save_path, index=False)
        # 这里的 dim 应该是 30*5 + 2(idx, label) = 152
        print(f"\n   ✅ 保存: {tag} - shape {res_df.shape} (预期 ~152 cols)")

if __name__ == "__main__":
    build_features()