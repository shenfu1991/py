import os
import pandas as pd

# 指定文件夹路径
folder_path = '//Users/xuanyuan/Documents/ls/'

# 获取文件夹内所有CSV文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 处理每个CSV文件
for csv_file in csv_files:
    # 读取CSV文件
    csv_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(csv_path)
    
    # 删除重复行
    df.drop_duplicates(inplace=True)
    
    # 保存处理后的CSV文件
    processed_csv_path = os.path.join(folder_path, f"processed_{csv_file}")
    df.to_csv(processed_csv_path, index=False)

print("All files have been processed.")
