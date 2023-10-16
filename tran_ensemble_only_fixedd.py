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
data_path = '/Users/xuanyuan/py/merged_pro-2.csv'  # Replace with your actual path
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

params = {
    'booster': 'gbtree',
    'objective': 'multi:softprob',
    'num_class': 4,
    # 'max_depth': 10,
    'random_state': 42,
}

print(params)

# Train XGBoost with the parameters
xgb_clf = XGBClassifier(**params)
from sklearn.ensemble import BaggingClassifier

# Using Bagging with XGBoost as the base classifier
bag_clf = BaggingClassifier(
    xgb_clf,
    n_estimators=10,
    max_samples=0.8,
    bootstrap=True,
    n_jobs=-1
)
bag_clf.fit(X_train_scaled, y_train)  # Training the Bagging classifier
            # early_stopping_rounds=10, 
            # eval_set=[(X_val_scaled, y_val)])

# Predict on the validation set
y_pred = bag_clf.predict(X_val_scaled)

# Print the classification report
print(classification_report(y_val, y_pred))
print(data_path)

# Save the model, scaler and label encoder to a .pkl file
with open("bagging_model.pkl", "wb") as pkl_file:
    pickle.dump({'model': bag_clf, 'scaler': scaler, 'label_encoder': label_encoder}, pkl_file)

    # 获取当前时间
end_time = datetime.now()

# 计算并打印执行时间
execution_time = end_time - start_time
print("脚本执行耗时: ",execution_time, "秒")


