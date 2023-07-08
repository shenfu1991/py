import os
import pandas as pd

# 设定文件夹路径
folder_path = '/Users/xuanyuan/Documents/ls/'

# 遍历文件夹内的所有.csv文件
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)

        # 读取csv文件
        df = pd.read_csv(file_path)

        # 将'result'列的'long'和'LN'替换为'L'，'short'和'SN'替换为'S'
        df['result'] = df['result'].replace(['LN'], 'long')
        df['result'] = df['result'].replace(['SN'], 'short')

        # 保存新的csv文件
        new_file_path = os.path.join(folder_path, 'new_' + filename)
        df.to_csv(new_file_path, index=False)

print("处理完成！")
