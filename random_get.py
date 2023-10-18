import pandas as pd
import os
import glob

# 指定包含CSV文件的文件夹路径
path = '/Users/xuanyuan/Documents/csv-sm/n'
all_files = glob.glob(os.path.join(path, "*.csv"))

list_of_dfs = []

# 遍历并读取每个CSV文件
for filename in all_files:
    df = pd.read_csv(filename)
    # 随机抽取10%的数据
    sampled_df = df.sample(frac=0.1)
    list_of_dfs.append(sampled_df)

# 合并所有的DataFrame
merged_df = pd.concat(list_of_dfs, ignore_index=True)

# 保存合并后的DataFrame为一个新的CSV文件
merged_df.to_csv('merged_sampled.csv', index=False)
