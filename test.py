import pandas as pd
from joblib import load
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# 读取新的csv文件
new_data_path = 'processed_n_test_15m.csv'  # 替换为你的新csv文件路径
new_df = pd.read_csv(new_data_path)

# 准备数据
features = ['current','avg','open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']
new_X = new_df[features]
new_y = new_df['result']

# 加载模型和标签编码器
best_model = load('model_15m_minv2.joblib')
le = load('label_encoder_15m_minv2.joblib')

#v7 对数据进行标准化
scaler = StandardScaler()
new_X = scaler.fit_transform(new_X)

# 使用之前的标签编码器转化标签
new_y = le.transform(new_y)

# 预测
new_y_pred = best_model.predict(new_X)

# 输出分类报告
print(classification_report(new_y, new_y_pred, target_names=le.classes_))
