import os
import pandas as pd

# 定义列名
column_names = ["timestamp","current",'open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']

# 指定文件夹路径
folder_path = "/Users/xuanyuan/Downloads/5mv2"

# 遍历文件夹内所有的csv文件
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        
        # 读取csv文件，并指定列名
        df = pd.read_csv(file_path, header=None, names=column_names)
        
        # 将带有新表头的DataFrame写回csv
        df.to_csv(file_path, index=False)
