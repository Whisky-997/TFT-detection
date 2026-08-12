import pandas as pd
import numpy as np
import os

def convert_to_tft_tensor():
    # 配置: 仅保留 32-16 的 TFT Tensor 生成
    # TFT 输入通常需要序列化: [Batch, Time_Steps, Features]
    # 注意: 这里的 Time_Steps 是指 "窗口的序列"。
    # 例如: 过去 32 个 "窗口统计特征" 预测 下一个 "窗口标签"。
    
    configs = [
        {
            "name": "Wavelet (150 dim)",
            "suffixes": ["_mean", "_std", "_max", "_min", "_energy"], 
            "out_dir": "/root/autodl-tmp/graduate-thesis/data/tensor_T32"
        }
    ]
    
    label_field = "window_label"
    T_SEQ_LEN = 32 # TFT的时间步长 (Sequence Length)
    
    for cfg in configs:
        print(f"\n🚀 开始生成 Tensor: {cfg['name']}")
        os.makedirs(cfg['out_dir'], exist_ok=True)
        
        for subset in ["train", "val", "test"]:
            # 读取标准化后的文件
            csv_path = os.path.join("/root/autodl-tmp/graduate-thesis/data/standardized", f"cic17_W32_S16_{subset}.csv")
            if not os.path.exists(csv_path): 
                print(f"   ⚠️ 跳过 {subset} (文件未找到)"); continue
            
            df = pd.read_csv(csv_path).sort_values("aux_window_index")
            
            # 筛选特征
            feat_cols = [c for c in df.columns if any(c.endswith(s) for s in cfg['suffixes'])]
            print(f"   - {subset}: 特征数 {len(feat_cols)}, 总行数 {len(df)}")
            
            X = df[feat_cols].values.astype(np.float32)
            y = df[label_field].values.astype(np.int32)
            
            # 构建序列
            # 逻辑: 取 i 到 i+32 作为输入 X，取 i+31 (当前) 或 i+32 (下一刻) 作为标签
            # 这里的步长(stride)设为1或者8都可以，取决于想要多少样本
            STRIDE = 1 
            
            X_seq, y_seq = [], []
            # 保证索引不越界
            for i in range(0, len(X) - T_SEQ_LEN + 1, STRIDE):
                X_seq.append(X[i : i + T_SEQ_LEN])
                # 标签取序列最后一个时间点的标签 (检测当前窗口序列末端是否异常)
                y_seq.append(y[i + T_SEQ_LEN - 1])
                
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            np.save(os.path.join(cfg['out_dir'], f"cic17_W32_S16_{subset}_X_T32.npy"), X_seq)
            np.save(os.path.join(cfg['out_dir'], f"cic17_W32_S16_{subset}_y_T32.npy"), y_seq)
            print(f"   ✅ 保存 {subset}: X shape {X_seq.shape}")

if __name__ == "__main__":
    convert_to_tft_tensor()