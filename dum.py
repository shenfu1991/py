import os
import pandas as pd

# 指定文件夹路径
folder_path = "/Users/xuanyuan/py/4h"

# 遍历文件夹内所有的csv文件
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        
        # 读取csv文件
        df = pd.read_csv(file_path)
        
        # 删除相邻的重复行
        df.drop_duplicates(inplace=True)
        
        # 将处理后的数据写回csv
        df.to_csv(file_path, index=False)
