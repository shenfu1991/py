import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump
import matplotlib.pyplot as plt
from datetime import datetime

# 获取当前时间
start_time = datetime.now()

# 打印当前时间
print("当前时间:", start_time)

path = 'processed_merged_1ha1.csv'

print(path)

# 读取数据
df = pd.read_csv(path)

# 准备数据
features = ['current','avg','open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']
X = df[features]
y = df['result']

# Assign higher weights to 'long' and 'short' classes
weights = np.ones(len(y))
weights[(y == 'long') | (y == 'short')] = 10
weights[(y == 'LN') | (y == 'SN')] = 1

# Label encoding
le = LabelEncoder()
y = le.fit_transform(y)
dump(le, 'label_encoder_.joblib') # save the label encoder

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

# Define model parameters for random search
param_distributions = {
    'objective': ['multi:softmax'],
    'num_class': [4],
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.5, 1]
}

# Perform Random Search
random_search = RandomizedSearchCV(estimator=xgb.XGBClassifier(eval_metric='mlogloss'), 
                                   param_distributions=param_distributions, 
                                   n_iter=20, 
                                   cv=3, 
                                   scoring='accuracy', 
                                   random_state=42)
random_search.fit(X_train, y_train, sample_weight=weights_train)

# Extract Best Model
best_model = random_search.best_estimator_

# Output Training and Testing accuracy
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

print('Training accuracy: ', accuracy_score(y_train, y_pred_train))
print('Testing accuracy: ', accuracy_score(y_test, y_pred_test))

# 保存模型
dump(best_model, 'model_.joblib')

# Print Classification Report
print(classification_report(y_test, y_pred_test, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix: \n", cm)

# Plot Feature Importance
xgb.plot_importance(best_model)
plt.show()


# 获取当前时间
end_time = datetime.now()

# 计算并打印执行时间
execution_time = end_time - start_time
print("脚本执行耗时: ",execution_time, "秒")
