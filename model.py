import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from joblib import load

# Load the model and the label encoder
xgb_best = load('model_.joblib')
le = load('label_encoder.joblib')

# Load the new data
data_new = pd.read_csv('RDNTUSDT_15m_15m_t.csv')  # replace 'new_data.csv' with your new csv file

# Select the top 3 features
#top_features = ["current", "signal", "avg"]  # replace this with your top features
top_features = ['current','avg','open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']

X_new_top_features = data_new[top_features]

# Make predictions
y_pred = xgb_best.predict(X_new_top_features)

# If 'result' is the column with the actual labels in the new csv file
y_true = le.transform(data_new['result'])

# Decode the predictions
y_pred_decoded = le.inverse_transform(y_pred)

# Print the classification report
print(classification_report(le.inverse_transform(y_true), le.inverse_transform(y_pred)))

# Print the confusion matrix
print(confusion_matrix(le.inverse_transform(y_true), le.inverse_transform(y_pred)))

# Save the predictions to a CSV file
pd.DataFrame(y_pred_decoded, columns=['Predictions']).to_csv('predictions.csv', index=False)
