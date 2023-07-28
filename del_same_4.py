import os
import pandas as pd

# 用于检查和删除具有相同特定列值的相邻行的函数
def check_and_drop_duplicates(df):
    cols_to_check = ['iRank','minRate','maxRate', 'volatility', 'sharp', 'signal']
    df = df.loc[~(df[cols_to_check].shift() == df[cols_to_check]).all(axis=1)]
    return df


path = "/Users/xuanyuan/Documents/ty"

# 遍历文件夹及其子文件夹中的所有CSV文件
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 检查和删除具有相同特定列值的相邻行
            df = check_and_drop_duplicates(df)
            
            # 将处理后的数据帧保存回CSV文件
            df.to_csv(file_path, index=False)

print('done')