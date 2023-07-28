import os
import pandas as pd

# 文件夹路径
folder_path = "/Users/xuanyuan/Downloads/7-12"

# 需要删除的列
columns_to_drop = ["open", "high", "low", "rate", "volume"]

# 遍历文件夹中的文件，包括所有子文件夹
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        # 检查是否是 CSV 文件
        if filename.endswith(".csv"):
            # 文件的完整路径
            file_path = os.path.join(root, filename)
            # 读取 CSV 文件
            df = pd.read_csv(file_path)
            # 删除指定的列
            df = df.drop(columns=columns_to_drop, errors='ignore')
            # 保存修改后的数据到文件
            df.to_csv(file_path, index=False)

print('done')