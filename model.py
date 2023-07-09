import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from joblib import load

# Load the model
xgb_best = load('model_.joblib')

# Load the label encoder
le = load('label_encoder.joblib')

# Load the new data
data = pd.read_csv('new_data.csv') # replace this with your new csv file

# Select the top 3 features
top_features = ["current", "signal", "avg"] # replace this with your top features
X_new = data[top_features]

# Make predictions on the new data
new_preds = xgb_best.predict(X_new)

# Decode the predictions
new_preds_decoded = le.inverse_transform(new_preds)

print("Predictions:", new_preds_decoded)
