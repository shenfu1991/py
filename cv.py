import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump

from datetime import datetime

# # 获取当前时间
start_time = datetime.now()

# 读取数据
df = pd.read_csv('/Users/xuanyuan/py/merged_3m.csv')

# 准备数据
features = ['open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']
X = df[features]
y = df['result']

# Label encoding
le = LabelEncoder()
y = le.fit_transform(y)
dump(le, 'label_encoder.joblib') # save the label encoder

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化模型
model = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=2, min_samples_split=5)

# 交叉验证
scores = cross_val_score(model, X_train, y_train, cv=5)

# 输出交叉验证的分数
print('Cross-validation scores: ', scores)

# 训练模型
model.fit(X_train, y_train)

# 保存模型
dump(model, 'model.joblib')


# # 获取当前时间
end_time = datetime.now()


# 打印当前时间
print("结束时间:", end_time)


# 计算并打印执行时间
execution_time = end_time - start_time
print("脚本执行耗时: ",execution_time, "秒")
