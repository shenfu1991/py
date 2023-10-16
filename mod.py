import os
import pandas as pd

# 指定包含CSV文件的文件夹路径
folder_path = '/Users/xuanyuan/Documents/csv-t'

# 遍历文件夹，找到所有CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # 完整的CSV文件路径
        full_path = os.path.join(folder_path, filename)
        
        # 读取CSV文件到DataFrame
        df = pd.read_csv(full_path)
        
        # 检查是否存在'upDownMa23'列
        if 'upDownMa23' in df.columns:
            # 修改'upDownMa23'列的值
            df['upDownMa23'] = df['upDownMa23'].apply(lambda x: 1 if x == 'up' else (0 if x == 'down' else x))
            
            # 保存修改后的DataFrame回同一个CSV文件
            df.to_csv(full_path, index=False)
