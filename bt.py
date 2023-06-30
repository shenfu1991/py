import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# 读取数据
df = pd.read_csv('/Users/xuanyuan/py/merged_30mv2.csv')

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
model = xgb.XGBClassifier()

# 设置参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [6, 10, 15],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.5, 0.7, 1.0]
}

# 初始化 Grid Search 对象
grid_search = GridSearchCV(model, param_grid, cv=5)

# 执行 Grid Search
grid_search.fit(X_train, y_train)

# 输出最优参数
print('Best Parameters: ', grid_search.best_params_)

# 使用最优参数训练模型
best_model = xgb.XGBClassifier(**grid_search.best_params_)
best_model.fit(X_train, y_train)

# 输出训练准确率
print('Training accuracy: ', best_model.score(X_train, y_train))

# 保存模型
dump(best_model, 'model.joblib')
