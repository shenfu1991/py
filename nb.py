import pandas as pd
from sklearn.utils import resample

# 读取CSV文件
df = pd.read_csv('merged_15mv7.csv')

# 对每一个结果值进行重采样
resampled = pd.DataFrame()
unique_results = df['result'].unique()
for result in unique_results:
    # 提取出该结果值的所有数据
    df_result = df[df['result'] == result]
    # 进行重采样，使其数量等于最大的那个结果值的数量
    df_result_resampled = resample(df_result, replace=True, 
                                   n_samples=df['result'].value_counts().max(), 
                                   random_state=123)
    # 添加到新的DataFrame中
    resampled = pd.concat([resampled, df_result_resampled])

# 保存新的DataFrame为CSV文件
resampled.to_csv('resampled.csv', index=False)
