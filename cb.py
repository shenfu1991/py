import pandas as pd

# 读取csv文件
df = pd.read_csv('merged_1h.csv')

counts = df['result'].value_counts()
max_count = counts[counts.index != 'none'].max()
none_rows = df[df['result'] == 'none']
excess_nones = none_rows.iloc[max_count:]  # 这些是需要删除的行
df = df.drop(excess_nones.index)

df.to_csv('path_to_your_file.csv', index=False)
