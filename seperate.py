import os
import pandas as pd

# 指定包含CSV文件的文件夹路径
# folder_path = '/Users/xuanyuan/Documents/csv-big'
folder_path = '/Users/xuanyuan/Downloads/d/ls'


# 遍历文件夹，找到所有CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # 完整的CSV文件路径
        full_path = os.path.join(folder_path, filename)
        
        # 读取CSV文件到DataFrame
        df = pd.read_csv(full_path)
        
        # # 拆分DataFrame基于'result'列的值
        # df_ls = df[df['result'].isin(['long', 'short'])]
        # df_none = df[df['result'].isin(['SN', 'LN'])]
            # 拆分DataFrame基于'result'列的值
        df_ls = df[df['status'].isin(['long'])]
        df_none = df[df['status'].isin(['short'])]
        
        # 保存拆分后的DataFrame到新的CSV文件
        new_filename_ls = os.path.splitext(filename)[0] + '_ls.csv'
        new_filename_none = os.path.splitext(filename)[0] + '_none.csv'
        
        df_ls.to_csv(os.path.join(folder_path, new_filename_ls), index=False)
        df_none.to_csv(os.path.join(folder_path, new_filename_none), index=False)
