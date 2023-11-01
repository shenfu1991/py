import pandas as pd
import json

def process_csv_with_percentile(file_path):
    # 读取CSV文件到DataFrame
    df = pd.read_csv(file_path)

    # 初始化结果字典
    stats = {}

    # 需要统计的列名
    columns_to_stat = ['long', 'short', 'topR', 'bottomR', 'longR', 'shortR', 'earnE']

    # 对每一列进行统计
    for col in columns_to_stat:
        if col in df.columns:
            stats[col] = {
                'max': df[col].max(),
                'min': df[col].min(),
                'mean': df[col].mean(),
                '20th_percentile': df[col].quantile(0.2),
                '80th_percentile': df[col].quantile(0.8)
            }

 # 保存为JSON文件
    with open(json_output_path, 'w') as f:
        json.dump(stats, f, indent=4)

    return json.dumps(stats, indent=4)

# 使用方法：
# file_path = 'your_file.csv'  # 替换为你的CSV文件路径
# result = process_csv_with_percentile(file_path)
# print(result)


# 使用方法：
fileName = '/Users/xuanyuan/Downloads/d/ls/merged_'
interval = '3m'
side = 'short'
file_path = fileName+interval+'_'+side+'.csv'  # 替换为你的CSV文件路径
json_output_path = '/Users/xuanyuan/Downloads/d/'+interval+"_"+side+'.json'
print(file_path)
result = process_csv_with_percentile(file_path)
print(result)
