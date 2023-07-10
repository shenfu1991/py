import pandas as pd
import os

# 指定你的文件夹路径
folder_path = '/Users/xuanyuan/py/15n'

# 遍历文件夹中的每一个文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 删除result列中为空的行
        df = df[df['result'].notna()]
        
        # 将结果写回CSV文件
        df.to_csv(file_path, index=False)
