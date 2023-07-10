import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


from datetime import datetime

# 获取当前时间
start_time = datetime.now()

# 打印当前时间
print("当前时间:", start_time)


path = 'merged_all.csv'

print(path)

# 读取数据
df = pd.read_csv(path)

features = ['current','avg','open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']

X = df[features]
y = df['result']

# Label encoding
le = LabelEncoder()
y = le.fit_transform(y)
dump(le, 'label_encoder_.joblib') # save the label encoder

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用最优参数训练模型
best_model = xgb.XGBClassifier(n_estimators=100,max_depth=10)
best_model.fit(X_train, y_train)

# 输出训练准确率
print('Training accuracy: ', best_model.score(X_train, y_train))
print("Testing accuracy: ", best_model.score(X_test, y_test))

# 保存模型
dump(best_model, 'model_.joblib')

# Output Training and Testing accuracy
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)


# Print Classification Report
print(classification_report(y_test, y_pred_test, target_names=le.classes_))

