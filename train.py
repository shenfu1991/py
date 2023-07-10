import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from joblib import dump

# Load the data
data = pd.read_csv('merged_15n.csv')

# Initialize a label encoder
le = LabelEncoder()

# Encode the target variable
data['result'] = le.fit_transform(data['result'])

# Split the data into features (X) and target (y)
X = data.drop('result', axis=1)
y = data['result']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select the top 3 features
top_features = ["current", "signal", "avg"] # replace this with your top features

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
