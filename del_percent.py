import pandas as pd
import os

# 指定文件夹路径
folder_path = '/Users/xuanyuan/Downloads/7-8/15'

# 列出该文件夹内所有csv文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for file in csv_files:
    # 加载csv文件
    df = pd.read_csv(os.path.join(folder_path, file))
    
    # 删除前75%的数据，保留后25%
    df = df.tail(int(len(df)*0.12))
    
    # 将处理后的数据保存回csv文件
    df.to_csv(os.path.join(folder_path, file), index=False)
