import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 加载CSV数据
df = pd.read_csv('/Users/xuanyuan/py/merged_3m_mix.csv') # 请将'your_file.csv'替换为你的文件路径

# 对目标进行编码，因为它是分类任务
le = LabelEncoder()
df['result'] = le.fit_transform(df['result'])

# 定义特征和目标
features = ['open','high','low','rate','volume','volatility','sharp','signal']
target = ['result']

X = df[features]
y = df[target]

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义需要试验的参数
param_grid = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

# 创建一个随机森林分类器
clf = RandomForestClassifier(n_estimators=100)

# 创建GridSearchCV对象
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)

# 使用训练集数据训练模型
grid_search.fit(X_train, y_train.values.ravel())

# 打印出最佳参数
print('Best Parameters: ', grid_search.best_params_)

# 用最优参数重新训练模型
best_clf = grid_search.best_estimator_

# 评估模型
print("Training accuracy: ", best_clf.score(X_train, y_train))
print("Testing accuracy: ", best_clf.score(X_test, y_test))
