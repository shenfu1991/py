
# Load the data
# data_path = '/Users/xuanyuan/Downloads/o/merged_3.csv'  # Replace with your actual path

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Load the data
data_path = '/Users/xuanyuan/py/merged_30.csv'  # Replace with your actual path
print(data_path)
data = pd.read_csv(data_path)

# Convert target labels to numeric labels
label_encoder = LabelEncoder()
data['result'] = label_encoder.fit_transform(data['result'])

# Split the data into features and target
X = data.drop('result', axis=1)
y = data['result']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Data scaling
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)

params = {
    'booster': 'gbtree',
    'num_parallel_tree': 10,
    # 'subsample': 0.53,
    # 'colsample_bynode': 1.0,
    # 'learning_rate': 1,
    'objective': 'multi:softprob',
    'num_class': 4,
    # 'max_depth': 10,
    'random_state': 42,
}



# Train XGBoost with the parameters
xgb_clf = XGBClassifier(**params)
xgb_clf.fit(X_train, y_train, 
            early_stopping_rounds=10, 
            eval_set=[(X_val, y_val)])

# Predict on the validation set
y_pred = xgb_clf.predict(X_val)

# Print the classification report
print(classification_report(y_val, y_pred))
print(data_path)

# Save the model, scaler and label encoder to a .pkl file
with open("xgboost_model_xgb.pkl", "wb") as pkl_file:
    pickle.dump({'model': xgb_clf, 'label_encoder': label_encoder}, pkl_file)

