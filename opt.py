import pandas as pd
from sklearn.utils import resample

# 加载数据
data_file_path = 'merged_csv.csv'  # 请替换为你的文件路径
df = pd.read_csv(data_file_path)

# 设置每个类别的样本大小
sample_size_per_class = 100000

# 对每个类别进行随机抽样
df_SN_sample = df[df['result'] == 'SN'].sample(n=sample_size_per_class, random_state=42)
df_LN_sample = df[df['result'] == 'LN'].sample(n=sample_size_per_class, random_state=42)
df_short_sample = df[df['result'] == 'short'].sample(n=sample_size_per_class, replace=True, random_state=42)
df_long_sample = df[df['result'] == 'long'].sample(n=sample_size_per_class, replace=True, random_state=42)

# 合并抽样得到的数据
df_sample = pd.concat([df_SN_sample, df_LN_sample, df_short_sample, df_long_sample])

# 打乱数据顺序
df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)

# 检查新数据集的标签分布
label_distribution_sample = df_sample['result'].value_counts()
print("新数据集的标签分布：\n", label_distribution_sample)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# 数据预处理
X = df_sample.drop('result', axis=1)
y = df_sample['result']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# # 参数优化
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1]
}

grid_clf = GridSearchCV(XGBClassifier(), param_grid, cv=3)
grid_clf.fit(X_train, y_train)

# 输出最优参数
print("最优参数：", grid_clf.best_params_)

# 使用最优参数进行预测
best_clf = grid_clf.best_estimator_
y_pred = best_clf.predict(X_val)

# 性能评估
print("分类报告：\n", classification_report(y_val, y_pred, target_names=label_encoder.classes_))


from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures

# 特征工程：创建多项式特征
poly = PolynomialFeatures(2)
X_train_poly = poly.fit_transform(X_train)
X_val_poly = poly.transform(X_val)

# 创建多个模型
clf1 = XGBClassifier(learning_rate=0.1, max_depth=7, n_estimators=100)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf3 = SVC(kernel='linear', probability=True, random_state=42)

# 模型集成：投票分类器
ensemble_clf = VotingClassifier(estimators=[
    ('xgb', clf1), ('rf', clf2), ('svc', clf3)], voting='soft')

ensemble_clf.fit(X_train_poly, y_train)

# 性能评估
y_pred = ensemble_clf.predict(X_val_poly)
print("分类报告：\n", classification_report(y_val, y_pred, target_names=label_encoder.classes_))

