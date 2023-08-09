import os
import pandas as pd

def process_csv(file_path):
    # 1. 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 2. 检查是否有'result'列
    if 'result' not in df.columns:
        print(f"'result' column not found in {file_path}. Skipping.")
        return
    
    # 3. 计算每种结果的数量
    sn_count = len(df[df['result'] == 'SN'])
    ln_count = len(df[df['result'] == 'LN'])
    short_count = len(df[df['result'] == 'short'])
    long_count = len(df[df['result'] == 'long'])
    
    # 4. 调整每种结果的数量
    # 如果'SN'的数量多于'short'的数量，删除多余的'SN'行
    if sn_count > short_count:
        drop_count = sn_count - short_count
        drop_indices = df[df['result'] == 'SN'].index[:drop_count]
        df.drop(drop_indices, inplace=True)
    
    # 如果'short'的数量多于'SN'的数量，删除多余的'short'行
    elif short_count > sn_count:
        drop_count = short_count - sn_count
        drop_indices = df[df['result'] == 'short'].index[:drop_count]
        df.drop(drop_indices, inplace=True)
        
    # 如果'LN'的数量多于'long'的数量，删除多余的'LN'行
    if ln_count > long_count:
        drop_count = ln_count - long_count
        drop_indices = df[df['result'] == 'LN'].index[:drop_count]
        df.drop(drop_indices, inplace=True)
    
    # 如果'long'的数量多于'LN'的数量，删除多余的'long'行
    elif long_count > ln_count:
        drop_count = long_count - ln_count
        drop_indices = df[df['result'] == 'long'].index[:drop_count]
        df.drop(drop_indices, inplace=True)
    
    # 5. 保存处理后的CSV文件
    df.to_csv(file_path, index=False)

def process_directory(directory_path):
    # 遍历文件夹及其子文件夹
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.csv'):
                process_csv(os.path.join(root, file))

# 使用函数
directory_path = "/Users/xuanyuan/Downloads/1-1"
process_directory(directory_path)

print('done')

