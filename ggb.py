
# Load the data
# data_path = '/Users/xuanyuan/Downloads/o/merged_3.csv'  # Replace with your actual path

import pandas as pd
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


# Load the data
data_path = '/Users/xuanyuan/py/merged_5mls.csv'  # Replace with your actual path
print(data_path)
data = pd.read_csv(data_path)

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



# params = {
#     # 'booster': 'gbtree',
#     # 'num_parallel_tree': 100,
#     # 'subsample': 0.63,
#     # 'colsample_bynode': 1.0,
#     # 'learning_rate': 1,
#     'objective': 'multi:softprob',
#     'num_class': 4,
#     # 'max_depth': 10,
#     'random_state': 42,
# }

params = {
    'booster': 'gbtree',
    # 'num_parallel_tree': 10,
    # 'subsample': 0.53,
    # 'colsample_bynode': 1.0,
    # 'learning_rate': 1,
    'objective': 'multi:softprob',
    'num_class': 2,
    'max_depth': 10,
    'random_state': 42,
}

print(params)


# Train XGBoost with the parameters
xgb_clf = XGBClassifier(**params)
xgb_clf.fit(X_train_scaled, y_train, 
            early_stopping_rounds=10, 
            eval_set=[(X_val_scaled, y_val)])

import numpy as np

# Predict on the validation set
y_pred = xgb_clf.predict(X_val_scaled)

# If y_pred is two-dimensional (e.g., probabilities for each class), convert it to one-dimensional
if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
    y_pred = np.argmax(y_pred, axis=1)

# Now you can use the classification_report
print(classification_report(y_val, y_pred))

print(data_path)

# Save the model, scaler and label encoder to a .pkl file
with open("xgboost_model.pkl", "wb") as pkl_file:
    pickle.dump({'model': xgb_clf, 'scaler': scaler, 'label_encoder': label_encoder}, pkl_file)

    # 获取当前时间
end_time = datetime.now()

# 计算并打印执行时间
execution_time = end_time - start_time
print("脚本执行耗时: ",execution_time, "秒")


