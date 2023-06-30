import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# 读取数据
df = pd.read_csv('/Users/xuanyuan/py/merged_15mv2.csv')

# 准备数据
features = ['open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']
X = df[features]
y = df['result']

# Label encoding
le = LabelEncoder()
y = le.fit_transform(y)
dump(le, 'label_encoder_15mv2.joblib') # save the label encoder

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用最优参数训练模型
best_model = xgb.XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.3, subsample=1)
best_model.fit(X_train, y_train)

# 输出训练准确率
print('Training accuracy: ', best_model.score(X_train, y_train))
print("Testing accuracy: ", best_model.score(X_test, y_test))

# 保存模型
dump(best_model, 'model_15mv2.joblib')
