import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

from datetime import datetime

# 获取当前时间
start_time = datetime.now()

# 打印当前时间
print("当前时间:", start_time)

interval = "15m"

name = "_"+interval+"_"+interval

# path = '/Users/xuanyuan/Documents/ty/RDNTUSDT' + name + '.csv'

path = 'merged_cs.csv'


# path = 'merged_' + name + '.csv'

print(path)

data = pd.read_csv(path)

sm = SMOTE(random_state=42)
X, y = sm.fit_resample(data.drop('result', axis=1), data['result'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
print(classification_report(y_test, predictions))

modelName = 'model_' + name + '.pkl'

with open(modelName, 'wb') as file:
    pickle.dump(model, file)



# 获取当前时间
end_time = datetime.now()

# 计算并打印执行时间
execution_time = end_time - start_time
print("脚本执行耗时: ",execution_time, "秒")
