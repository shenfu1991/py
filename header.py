import os
import pandas as pd

# 指定你要查找的文件夹路径
directory = '/Users/xuanyuan/Downloads/all/15mv2'

# 需要检查的列名
columns = ['timestamp', 'current', 'open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']

# 列名转为一行字符串，用逗号分隔
header_line = ','.join(columns) + '\n'

# 遍历指定文件夹
for filename in os.listdir(directory):
    if filename.endswith(".csv"):  # 检查文件是否为csv格式
        file_path = os.path.join(directory, filename)

        with open(file_path, 'r+') as f:
            first_line = f.readline().strip()
            
            # 检查文件的第一行是否包含所有需要的列名
            if not all(col in first_line for col in columns):
                f.seek(0)  # 回到文件的开始
                content = f.read()  # 读取整个文件的内容
                
                # 将列名添加到文件的第一行
                f.seek(0)  # 再次回到文件的开始
                f.write(header_line + content)  # 先写入新的第一行，然后再写入原来的内容
                
                print(f"Updated file: {filename}")
            else:
                print(f"Skipped file: {filename}")
    else:
        print(f"Ignored file: {filename}")
