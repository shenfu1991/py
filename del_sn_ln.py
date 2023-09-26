import os
import pandas as pd

def process_csv_files(directory):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory):
        # 检查是否为CSV文件
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            
            # 读取CSV文件
            df = pd.read_csv(filepath)
            
            # 删除result列中值为SN或LN的行
            df = df[~df['result'].isin(['SN', 'LN'])]
            
            # 保存处理后的数据到新文件
            new_filename = filename.split('.')[0] + '_processed.csv'
            new_filepath = os.path.join(directory, new_filename)
            df.to_csv(new_filepath, index=False)

            print(f"Processed and saved {new_filename}")
            os.remove(filepath)

# 使用方法
directory = "/Users/xuanyuan/Documents/5mls"  # 例如：directory = "/path/to/your/csv_files"
print(directory)
process_csv_files(directory)
