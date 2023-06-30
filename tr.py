from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import pandas as pd

from datetime import datetime

# 获取当前时间
start_time = datetime.now()

# 打印当前时间
print("当前时间:", start_time)

# 加载数据
data = pd.read_csv('/Users/xuanyuan/py/merged_30mv2.csv')

# 创建并拟合 LabelEncoder
le = LabelEncoder()
y = le.fit_transform(data['result'])

# 训练模型
X = data[['open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(max_depth=20, min_samples_leaf=2, min_samples_split=5)
clf.fit(X_train, y_train)

# 评估模型
print("Training accuracy: ", clf.score(X_train, y_train))
print("Testing accuracy: ", clf.score(X_test, y_test))

# 保存模型和 LabelEncoder
dump(clf, 'model_30m.joblib')
dump(le, 'label_encoder_30m.joblib')


# 获取当前时间
end_time = datetime.now()

# 计算并打印执行时间
execution_time = end_time - start_time
print("脚本执行耗时: ",execution_time, "秒")
