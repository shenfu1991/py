import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from joblib import dump


from datetime import datetime

# 获取当前时间
start_time = datetime.now()

# 打印当前时间
print("当前时间:", start_time)

# Load the data
data = pd.read_csv('cleaned_merged_15n.csv')

# Initialize a label encoder
le = LabelEncoder()

# Encode the target variable
data['result'] = le.fit_transform(data['result'])

# Split the data into features (X) and target (y)
X = data.drop('result', axis=1)
y = data['result']

# Split the data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select the top 3 features
# features = ['current','avg','open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']
top_features = ["signal","volatility","current","avg","sharp"] # replace this with your top features


# 9      signal    0.136020
# 7  volatility    0.130679
# 0     current    0.130145
# 1         avg    0.112297
# 8       sharp    0.104097
# 6      volume    0.078732
# 3        high    0.078122
# 4         low    0.077723
# 2        open    0.076249
# 5        rate    0.075937

#       feature  importance
# 0     current    0.161568
# 1         avg    0.136167
# 9      signal    0.129838
# 7  volatility    0.122049
# 8       sharp    0.091632
# 3        high    0.075952
# 4         low    0.075866
# 2        open    0.074625
# 6      volume    0.066773
# 5        rate    0.065531

X_train_top_features = X_train[top_features]


# Train the XGBClassifier with the best parameters
xgb_best = XGBClassifier(max_depth=7,n_estimators=150,learning_rate=0.2, eval_metric='mlogloss', random_state=42)
xgb_best.fit(X_train_top_features, y_train)

# Save the model
dump(xgb_best, 'model_.joblib')

# Save the label encoder
dump(le, 'label_encoder.joblib')


from sklearn.metrics import classification_report

# Predict on the test set
y_pred = xgb_best.predict(X_test[top_features])

# Print the classification report
print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred)))


# 获取当前时间
end_time = datetime.now()

# 计算并打印执行时间
execution_time = end_time - start_time
print("脚本执行耗时: ",execution_time, "秒")