import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

def standardize_data_cic17():
    # 统计后缀
    stat_suffixes = ["_mean", "_std", "_max", "_min", "_energy"]
    core_fields = ["aux_window_index", "window_label"]
    
    base_dir = "/root/autodl-tmp/graduate-thesis/data"
    feature_dir = os.path.join(base_dir, "feature")
    output_dir = os.path.join(base_dir, "standardized")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "scalers"), exist_ok=True)
    
    scaler = MinMaxScaler((0, 1))
    
    # 1. 先读取 Train 数据来 Fit Scaler
    print("✅ 阶段1: 读取 Train 集并拟合 Scaler...")
    train_path = os.path.join(feature_dir, "cic2017_temporal_feature_W32_S16_train.csv")
    if not os.path.exists(train_path):
        print("❌ 训练集特征文件不存在"); return

    df_train = pd.read_csv(train_path)
    # 动态筛选特征列
    feat_cols = [c for c in df_train.columns if any(c.endswith(s) for s in stat_suffixes) and c not in core_fields]
    
    X_train = df_train[feat_cols].values.astype(np.float64)
    scaler.fit(X_train)
    joblib.dump(scaler, os.path.join(output_dir, "scalers/scaler_cic17_W32_S16_features.pkl"))
    print(f"   Scaler拟合完成，特征维度: {len(feat_cols)}")
    
    # 2. 对 Train, Val, Test 进行 Transform 并保存
    print("✅ 阶段2: 转换并保存全量数据...")
    for subset in ["train", "val", "test"]:
        path = os.path.join(feature_dir, f"cic2017_temporal_feature_W32_S16_{subset}.csv")
        if not os.path.exists(path): continue
        
        df = pd.read_csv(path)
        # 确保列顺序一致
        X = df[feat_cols].values.astype(np.float64)
        
        # 变换并截断 (Clip防止越界)
        df[feat_cols] = np.clip(scaler.transform(X), 0, 1)
        
        save_path = os.path.join(output_dir, f"cic17_W32_S16_{subset}.csv")
        df.to_csv(save_path, index=False)
        print(f"   ✅ {subset} 完成 -> {save_path}")

if __name__ == "__main__":
    standardize_data_cic17()