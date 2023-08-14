import os
import pandas as pd

def balance_data(df):
    # 获取short和long的数量
    short_count = len(df[df['result'] == 'short'])
    long_count = len(df[df['result'] == 'long'])

#最小
    minCount = long_count

    if short_count < long_count:
        minCount = short_count

    print(minCount)

    # 获取SN和LN的数量
    sn_count = len(df[df['result'] == 'SN'])
    ln_count = len(df[df['result'] == 'LN'])

    # 计算需要保留的SN和LN的数量
    required_sn_count = 10 * minCount
    required_ln_count = 10 * minCount

    # 如果short的数量超出所需数量，只保留所需的数量
    if short_count > minCount:
        drop_short_count = short_count - minCount
        short_indices = df[df['result'] == 'short'].index[-drop_short_count:]
        df = df.drop(short_indices)

    # 如果long的数量超出所需数量，只保留所需的数量
    if long_count > minCount:
        drop_long_count = long_count - minCount
        long_indices = df[df['result'] == 'long'].index[-drop_long_count:]
        df = df.drop(long_indices)

    # 如果SN或LN的数量超出所需数量，只保留所需的数量
    if sn_count > required_sn_count:
        drop_sn_count = sn_count - required_sn_count
        sn_indices = df[df['result'] == 'SN'].index[-drop_sn_count:]
        df = df.drop(sn_indices)
    
    if ln_count > required_ln_count:
        drop_ln_count = ln_count - required_ln_count
        ln_indices = df[df['result'] == 'LN'].index[-drop_ln_count:]
        df = df.drop(ln_indices)

    return df

def process_all_csv_in_folder(folder_path):
    # 递归地遍历文件夹，找到所有的 CSV 文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 使用balance_data函数处理数据
                balanced_df = balance_data(df)
                
                # 保存处理后的CSV文件
                balanced_df.to_csv(file_path, index=False)
                print(f"Processed {file_path}")



process_all_csv_in_folder("/Users/xuanyuan/Documents/10")
