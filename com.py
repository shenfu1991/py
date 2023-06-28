import pandas as pd
import glob

# 设置csv文件的路径
path = r'/Users/xuanyuan/Documents/mix-30m' # 只需要修改此处为你的目录路径

# 获取所有csv文件
all_files = glob.glob(path + "/*.csv")

# 创建一个空列表，用于存放各个dataframe
li = []

# 遍历所有csv文件，并读取内容
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

# 使用pandas.concat函数将所有的dataframe合并到一个dataframe中
frame = pd.concat(li, axis=0, ignore_index=True)

# 如果需要，可以将合并后的数据保存为新的CSV文件
frame.to_csv("merged.csv", index=False)
