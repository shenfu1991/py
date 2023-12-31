import pandas as pd
from sklearn.utils import resample
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from datetime import datetime

# 获取当前时间
start_time = datetime.now()

# 打印当前时间
print("当前时间:", start_time)

# 加载数据
data_file_path = 'merged_mix.csv'  # 请替换为你的文件路径
df = pd.read_csv(data_file_path)
print(data_file_path)



import csv
from collections import Counter

# 定义一个函数来读取 CSV 文件并统计目标列的值
def count_target_values(csv_file_path, target_column):
    counts = Counter()
    
    with open(csv_file_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            target_value = row[target_column]
            counts[target_value] += 1
            
    return counts


counts = count_target_values(data_file_path, 'result')

# 打印结果
for value, count in counts.items():
    print(f"{value}: {count}")







# 设置每个类别的样本大小
sample_size_per_class_SN_LN = 1500000
sample_size_per_class = 300000

# 对每个类别进行随机抽样
df_SN_sample = df[df['result'] == 'SN'].sample(n=sample_size_per_class_SN_LN, random_state=42)
df_LN_sample = df[df['result'] == 'LN'].sample(n=sample_size_per_class_SN_LN, random_state=42)
df_short_sample = df[df['result'] == 'short'].sample(n=sample_size_per_class, replace=True, random_state=42)
df_long_sample = df[df['result'] == 'long'].sample(n=sample_size_per_class, replace=True, random_state=42)

# 合并抽样得到的数据
df_sample = pd.concat([df_SN_sample, df_LN_sample, df_short_sample, df_long_sample])

# 打乱数据顺序
data = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)

# 检查新数据集的标签分布
label_distribution_sample = data['result'].value_counts()
print("新数据集的标签分布：\n", label_distribution_sample)

# Convert target labels to numeric labels
label_encoder = LabelEncoder()
data['result'] = label_encoder.fit_transform(data['result'])

# Split the data into features and target
X = data.drop('result', axis=1)
y = data['result']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

params = {
    'n_estimators': 10000,
    'learning_rate': 0.1,
    'max_depth': 7,
    'early_stopping_rounds': 10
}

print(params)

# Train XGBoost with the parameters
xgb_clf = XGBClassifier(**params)
xgb_clf.fit(X_train_scaled, y_train,eval_set=[(X_val_scaled, y_val)])

# Predict on the validation set
y_pred = xgb_clf.predict(X_val_scaled)

# Print the classification report
print(classification_report(y_val, y_pred))

# Save the model, scaler and label encoder to a .pkl file
with open("xgboost_model2.pkl", "wb") as pkl_file:
    pickle.dump({'model': xgb_clf, 'scaler': scaler, 'label_encoder': label_encoder}, pkl_file)

    # 获取当前时间
end_time = datetime.now()

# 计算并打印执行时间
execution_time = end_time - start_time
print("脚本执行耗时: ",execution_time, "秒")