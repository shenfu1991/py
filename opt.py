import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

from datetime import datetime

# 获取当前时间
start_time = datetime.now()

# 打印当前时间
print("当前时间:", start_time)

# 加载数据
data_file_path = 'merged_csv.csv'  # 请替换为你的文件路径
df = pd.read_csv(data_file_path)
# 假设 df 是一个 Pandas DataFrame

# 设置每个类别的样本大小
sample_size_per_class = 100000

# 对每个类别进行随机抽样
df_SN_sample = df[df['result'] == 'SN'].sample(n=sample_size_per_class, random_state=42)
df_LN_sample = df[df['result'] == 'LN'].sample(n=sample_size_per_class, random_state=42)
df_short_sample = df[df['result'] == 'short'].sample(n=sample_size_per_class, replace=True, random_state=42)
df_long_sample = df[df['result'] == 'long'].sample(n=sample_size_per_class, replace=True, random_state=42)

# 合并抽样得到的数据
df_sample = pd.concat([df_SN_sample, df_LN_sample, df_short_sample, df_long_sample])

# 打乱数据顺序
df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)

# 检查新数据集的标签分布
label_distribution_sample = df_sample['result'].value_counts()
print("新数据集的标签分布：\n", label_distribution_sample)




# 特征交叉
df_sample['rsi_so'] = df_sample['rsi'] * df_sample['so']
df_sample['mfi_cci'] = df_sample['mfi'] * df_sample['cci']

# 数据预处理
X = df_sample.drop('result', axis=1)
y = df_sample['result']

# 标签编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 分割数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 使用随机森林模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 性能评估
# 性能评估
y_pred = clf.predict(X_val)
target_names_str = [str(cls) for cls in label_encoder.classes_]
print("分类报告：\n", classification_report(y_val, y_pred, target_names=target_names_str))

# Save the model, scaler and label encoder to a .pkl file
with open("xgboost_model.pkl", "wb") as pkl_file:
    pickle.dump({'model': clf, 'scaler': scaler, 'label_encoder': label_encoder}, pkl_file)

    # 获取当前时间
end_time = datetime.now()

# 计算并打印执行时间
execution_time = end_time - start_time
print("脚本执行耗时: ",execution_time, "秒")