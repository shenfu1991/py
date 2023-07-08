import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from joblib import load
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


from datetime import datetime

# 获取当前时间
start_time = datetime.now()

# 打印当前时间
print("当前时间:", start_time)


test_path = '/Users/xuanyuan/Documents/new_GRTUSDT_15m_.csv'  # 提供测试数据的路径

print(test_path)

# 读取测试数据
df_test = pd.read_csv(test_path)

features = ['current','avg','open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']

X_test = df_test[features]
y_test = df_test['result']

# 加载之前保存的标签编码器
le = load('label_encoder_15v33.joblib') 
y_test = le.transform(y_test)  # 对标签进行编码

# 加载训练好的模型
best_model = load('model_15v33.joblib')

# 输出测试准确率
print("Testing accuracy: ", best_model.score(X_test, y_test))

# Output Testing accuracy
y_pred_test = best_model.predict(X_test)

# Print Classification Report
print(classification_report(y_test, y_pred_test, target_names=le.classes_))

# 获取当前时间
end_time = datetime.now()

# 计算并打印执行时间
execution_time = end_time - start_time
print("脚本执行耗时: ",execution_time, "秒")
