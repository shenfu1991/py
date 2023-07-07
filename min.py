import pandas as pd
import os

# 设定要处理的文件夹路径
folder_path = '/Users/xuanyuan/Documents/ui/'

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        
        # 读取csv文件
        df = pd.read_csv(file_path)
        
        # 获取result列中出现次数最少的值
        min_count = df['result'].value_counts().min()

        # 根据最小值对其他值进行处理
        new_df = pd.DataFrame()

        for value in df['result'].unique():
            temp_df = df[df['result'] == value][:min_count]
            new_df = pd.concat([new_df, temp_df])

        # 将处理后的DataFrame保存为新的csv文件
        new_file_path = os.path.join(folder_path, f'processed_{filename}')
        new_df.to_csv(new_file_path, index=False)
