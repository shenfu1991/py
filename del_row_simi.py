import pandas as pd
import os

# 获取文件夹内所有csv文件的路径
folder_path = '/Users/xuanyuan/Documents/p'  # 替换成你的文件夹路径

# 遍历文件夹中的每一个文件
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):  # 确保只处理CSV文件
        file_path = os.path.join(folder_path, filename)
        # 读取CSV文件到DataFrame
        df = pd.read_csv(file_path)

        # 通过将相邻两行的"volatility", "sharp", "signal"列的值进行比较来判断是否相等
        # 如果这些列的值相等，则标记为True，否则为False
        duplicate_mask = (df['volatility'].eq(df['volatility'].shift()) &
                          df['sharp'].eq(df['sharp'].shift()) &
                          df['signal'].eq(df['signal'].shift()))
        
        # 将标记为True的行（也就是重复的行）从DataFrame中删除
        df = df.loc[~duplicate_mask]

        # 将处理过的DataFrame保存到新的CSV文件中
        new_file_path = os.path.join(folder_path, "processed_" + filename)
        df.to_csv(new_file_path, index=False)
