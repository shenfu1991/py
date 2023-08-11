import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
import pickle
from datetime import datetime

# Getting current time
start_time = datetime.now()
print("当前时间:", start_time)

interval = "15m"
name = "_"+interval+"_"+interval
path = '/Users/xuanyuan/Downloads/o/merged_3.csv'
print(path)

# Load the data
data = pd.read_csv(path)

# Display the value counts for the 'result' column
print(data['result'].value_counts())

# Handling class imbalance by oversampling the minority classes
# Assuming 'long' and 'short' are the minority classes based on the problem description
long_data = data[data['result'] == 'long']
short_data = data[data['result'] == 'short']
ln_data = data[data['result'] == 'LN']
sn_data = data[data['result'] == 'SN']

# Oversampling the 'long' and 'short' classes
long_upsampled = resample(long_data, replace=True, n_samples=len(ln_data), random_state=42)
short_upsampled = resample(short_data, replace=True, n_samples=len(sn_data), random_state=42)

# Combining the upsampled data
data_upsampled = pd.concat([ln_data, sn_data, long_upsampled, short_upsampled])

# Split the data into training and test sets
X = data_upsampled.drop('result', axis=1)
y = data_upsampled['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier with increased number of trees
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))

# Print feature importances
feature_importances = pd.DataFrame(clf.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
print("Feature Importances:")
print(feature_importances)

# Save the model to a .pkl file
modelName = 'model_' + name + '.pkl'
with open(modelName, 'wb') as file:
    pickle.dump(clf, file)

# Getting current time
end_time = datetime.now()

# Calculate and print execution time
execution_time = end_time - start_time
print("脚本执行耗时: ",execution_time, "秒")
