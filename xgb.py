
# Load the data
# data_path = '/Users/xuanyuan/Downloads/o/merged_3.csv'  # Replace with your actual path

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Load the data
data_path = '/Users/xuanyuan/Downloads/1-1/merged_4.csv'  # Replace with your actual path
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
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Set parameters for XGBoost
params = {
    'objective': 'multi:softmax',
    'num_class': 4,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 200,
    'random_state': 42,
    'eval_metric': 'mlogloss'
}

# Train XGBoost with the parameters
xgb_clf = XGBClassifier(**params)
xgb_clf.fit(X_train_scaled, y_train, 
            early_stopping_rounds=10, 
            eval_set=[(X_val_scaled, y_val)])

# Predict on the validation set
y_pred = xgb_clf.predict(X_val_scaled)

# Print the classification report
print(classification_report(y_val, y_pred))

# Save the model, scaler and label encoder to a .pkl file
with open("xgboost_model.pkl", "wb") as pkl_file:
    pickle.dump({'model': xgb_clf, 'scaler': scaler, 'label_encoder': label_encoder}, pkl_file)

