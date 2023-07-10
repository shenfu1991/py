import pandas as pd
import os

# 获取文件夹内所有csv文件的路径
folder_path = '/Users/xuanyuan/Documents/n'  # 替换成你的文件夹路径
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 定义一个函数用于删除两行完全相同的行
def drop_identical_rows(df):
    # 在这里，我们将这四列作为子集，检查它们是否在连续的行中完全相同
    subset = ['volume', 'volatility', 'sharp', 'signal']
    df.drop_duplicates(subset=subset, keep=False, inplace=True)
    return df

# 对每个csv文件进行处理
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)

    # 删除完全相同的行
    df = drop_identical_rows(df)

    # 将结果保存为新的csv文件
    new_file_path = os.path.join(folder_path, 'new_'+file)
    df.to_csv(new_file_path, index=False)

print('处理完成！')
