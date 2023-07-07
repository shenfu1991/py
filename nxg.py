import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from joblib import dump
from datetime import datetime

# 获取当前时间
start_time = datetime.now()

# 打印当前时间
print("当前时间:", start_time)

path = '/Users/xuanyuan/py/resampled.csv'

print(path)

# 读取数据
df = pd.read_csv(path)

# 准备数据
#features = ['open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']
features = ['current','avg','open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']
#
X = df[features]
y = df['result']

# 特征标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Label encoding
le = LabelEncoder()
y = le.fit_transform(y)
dump(le, 'label_encoder_.joblib') # save the label encoder

# 过采样处理
smote = SMOTE(random_state=0)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 使用最优参数训练模型
best_model = xgb.XGBClassifier(n_estimators=100, max_depth=15, learning_rate=0.1, subsample=0.5)
best_model.fit(X_train, y_train)

# 输出训练准确率
print('Training accuracy: ', best_model.score(X_train, y_train))
print("Testing accuracy: ", best_model.score(X_test, y_test))

# 保存模型
dump(best_model, 'model_.joblib')

# 获取当前时间
end_time = datetime.now()

# 计算并打印执行时间
execution_time = end_time - start_time
print("脚本执行耗时: ",execution_time, "秒")


# # 加载必要的库
# from joblib import load
# from sklearn.metrics import classification_report
# import pandas as pd


# # 读取你的新csv文件
# new_data_path = 'merged.csv'  # 替换为你的新csv文件路径
# new_df = pd.read_csv(new_data_path)

# # 准备数据
# new_X = new_df[features]
# new_y = new_df['result']

# # 对数据进行相同的预处理
# new_X = scaler.transform(new_X)  # 注意这里使用transform, 不是fit_transform
# new_y = le.transform(new_y)  # 用之前保存的label encoder转化

# # 加载模型
# best_model = load('model_4hv9.joblib.joblib')

# # 预测
# new_y_pred = best_model.predict(new_X)

# # 输出分类报告，这个报告包含了各个标签的精确度、召回率等信息
# print(classification_report(new_y, new_y_pred, target_names=le.classes_))

