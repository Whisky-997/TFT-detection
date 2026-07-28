# 文件名：02_clean_cic2017.py
# 策略升级：20分块 + 7:2:1 划分
import pandas as pd
import numpy as np
import os

def load_and_split_interleaved_v2(cic2017_raw_dir):
    """
    【策略升级 V2】
    1. 细粒度分块：将每天数据切分为 20 块 (num_blocks=20)，提高攻击覆盖率。
    2. 黄金比例 7:2:1：
       - Train (70%): 14 块
       - Test  (20%): 4 块
       - Val   (10%): 2 块
    3. 均匀穿插：确保测试和验证集均匀分布在早中晚。
    """
    if not os.path.exists(cic2017_raw_dir):
        print(f"❌ 错误：目录 {cic2017_raw_dir} 不存在")
        return None, None, None
        
    csv_files = sorted([f for f in os.listdir(cic2017_raw_dir) if f.endswith('.csv')])
    
    train_pool = []
    val_pool = []
    test_pool = []
    
    # === 分块分配策略 (总共20块) ===
    # Test (4块): 索引 4, 9, 14, 19 (均匀分布在 20%, 45%, 70%, 95% 处)
    test_indices = {4, 9, 14, 19}
    
    # Val (2块): 索引 2, 12 (分布在 10%, 60% 处，错开 Test)
    val_indices = {2, 12}
    
    # Train (14块): 剩下的全是训练集
    # train_indices = {0, 1, 3, 5, 6, 7, 8, 10, 11, 13, 15, 16, 17, 18}
    
    print(f"✅ 开始处理 {len(csv_files)} 个原始文件 (20分块 + 7:2:1策略)...")
    
    for file in csv_files:
        file_path = os.path.join(cic2017_raw_dir, file)
        try:
            # 1. 读取 & 基础清洗
            df = pd.read_csv(file_path, encoding='latin-1')
            df = clean_basic_logic(df)
            n = len(df)
            
            # 忽略极小文件，防止分块报错
            if n < 2000: 
                print(f"⚠️ 跳过过小文件: {file}"); continue
            
            # 2. 计算分块大小 (20块)
            num_blocks = 20
            block_size = n // num_blocks
            
            print(f"  - {file} ({n}行) -> 切分20块...")
            
            # 3. 按块切分并分发
            for i in range(num_blocks):
                start = i * block_size
                # 最后一个块包含剩余所有数据 (防止除不尽丢失数据)
                end = (i + 1) * block_size if i < num_blocks - 1 else n
                
                block_data = df.iloc[start:end].copy()
                
                if i in test_indices:
                    test_pool.append(block_data)
                elif i in val_indices:
                    val_pool.append(block_data)
                else:
                    train_pool.append(block_data)
            
        except Exception as e:
            print(f"❌ 处理文件 {file} 失败: {e}")
            
    # 4. 合并所有池子
    final_train = pd.concat(train_pool, ignore_index=True)
    final_val = pd.concat(val_pool, ignore_index=True)
    final_test = pd.concat(test_pool, ignore_index=True)
    
    return final_train, final_val, final_test

def clean_basic_logic(df):
    """
    基础清洗逻辑 (保持不变)
    """
    df.columns = [col.strip() for col in df.columns]
    
    if 'Label' in df.columns:
        df['Label'] = df['Label'].astype(str).str.strip()
        df['Label'] = df['Label'].apply(lambda x: 0 if 'benign' in x.lower() else 1)
    
    useless = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'SimillarHTTP']
    df = df.drop(columns=[c for c in useless if c in df.columns], errors='ignore')
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.drop_duplicates()
    return df

def apply_winsorization_strict(train_df, val_df, test_df):
    """
    严防泄露：仅在 Train 上计算统计量
    """
    core_cols = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean'
    ]
    target_cols = [c for c in core_cols if c in train_df.columns]
    print(f"✅ 开始计算异常值阈值 (仅基于 Train 集)...")
    
    for col in target_cols:
        # 1. Log变换 (Apply to All)
        for d in [train_df, val_df, test_df]:
            d[col] = d[col].apply(lambda x: 0 if x < 0 else x)
            d[col] = np.log1p(d[col])
            
        # 2. 计算统计量 (Only Train)
        mu = train_df[col].mean()
        sigma = train_df[col].std()
        if sigma == 0: continue
        
        upper = mu + 5 * sigma
        lower = mu - 5 * sigma
        
        # 3. 盖帽 (Apply to All)
        for d in [train_df, val_df, test_df]:
            d[col] = d[col].clip(lower, upper)
            
    return train_df, val_df, test_df

if __name__ == "__main__":
    # 路径配置
    raw_dir = "/root/autodl-tmp/graduate-thesis/dataset/CICIDS2017_FULL"
    out_dir = "/root/autodl-tmp/graduate-thesis/data/cleaned"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. 执行 V2 版切分 (20块, 7:2:1)
    train_df, val_df, test_df = load_and_split_interleaved_v2(raw_dir)
    
    if train_df is None:
        print("❌ 处理失败"); exit()
        
    print(f"\n✅ 切分统计 (Total Samples):")
    print(f"   Train (70%): {len(train_df)}")
    print(f"   Test  (20%): {len(test_df)}")
    print(f"   Val   (10%): {len(val_df)}")
    
    # 2. 异常值处理
    train_df, val_df, test_df = apply_winsorization_strict(train_df, val_df, test_df)
    
    # 3. 保存结果
    train_df.to_csv(os.path.join(out_dir, "cic2017_cleaned_train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "cic2017_cleaned_val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "cic2017_cleaned_test.csv"), index=False)
    
    print("\n✅ 数据清洗完成 (策略V2：20分块 + 7:2:1)！")