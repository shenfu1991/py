import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

from datetime import datetime

# 获取当前时间
start_time = datetime.now()

# 打印当前时间
print("当前时间:", start_time)

interval = "15m"
name = "_"+interval+"_"+interval
# path = 'merged_cs.csv'
path = '/Users/xuanyuan/Downloads/4-1/merged_3.csv'
# path = "/Users/xuanyuan/Documents/1-1/merged_3.csv"
print(path)

# Load the data
data = pd.read_csv(path)

# Display the value counts for the 'result' column
print(data['result'].value_counts())

# Split the data into training and test sets
X = data.drop('result', axis=1)
y = data['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))

# Save the model to a .pkl file
modelName = 'model_' + name + '.pkl'
with open(modelName, 'wb') as file:
    pickle.dump(clf, file)


# 获取当前时间
end_time = datetime.now()

# 计算并打印执行时间
execution_time = end_time - start_time
print("脚本执行耗时: ",execution_time, "秒")
