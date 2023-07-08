import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump
import matplotlib.pyplot as plt
from datetime import datetime

# 获取当前时间
start_time = datetime.now()

# 打印当前时间
print("当前时间:", start_time)

path = 'processed_merged_15ma1.csv'

print(path)

# 读取数据
df = pd.read_csv(path)

# 准备数据
features = ['current','avg','open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']
X = df[features]
y = df['result']

# Assign higher weights to 'long' and 'short' classes
weights = np.ones(len(y))
weights[(y == 'long') | (y == 'short')] = 1
weights[(y == 'LN') | (y == 'SN')] = 1

# Label encoding
le = LabelEncoder()
y = le.fit_transform(y)
dump(le, 'label_encoder_.joblib') # save the label encoder

# No need for feature scaling
# X = X

# Split the data
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

# Define model parameters manually
params = {
    'n_estimators': 100,
    'max_depth': 10,
    'subsample': 1,
    'min_child_weight': 1,
    'n_jobs': -1
}

# Train the model
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, sample_weight=weights_train)

# Output Training and Testing accuracy
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print('Training accuracy: ', accuracy_score(y_train, y_pred_train))
print('Testing accuracy: ', accuracy_score(y_test, y_pred_test))

# 保存模型
dump(model, 'model_.joblib')

# Print Classification Report
print(classification_report(y_test, y_pred_test, target_names=le.classes_))

# 获取当前时间
end_time = datetime.now()

# 计算并打印执行时间
execution_time = end_time - start_time
print("脚本执行耗时: ",execution_time, "秒")
