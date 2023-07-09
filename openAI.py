import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from joblib import dump

# Load the data
data = pd.read_csv('RDNTUSDT_15m_15m.csv')

# Initialize a label encoder
le = LabelEncoder()

# Encode the target variable
data['result'] = le.fit_transform(data['result'])

# Split the data into features (X) and target (y)
X = data.drop('result', axis=1)
y = data['result']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a random forest classifier
rf = RandomForestClassifier(random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Find feature importances
importances = rf.feature_importances_

# Select the top 3 features
top_features = ["current", "signal", "avg"] # replace this with your top features
X_train_top_features = X_train[top_features]
X_test_top_features = X_test[top_features]

# Define the parameter grid
# param_grid = {
#     'max_depth': [3, 5, 7],
#     'n_estimators': [50, 100, 150],
#     'learning_rate': [0.01, 0.1, 0.2]
# }

param_grid = {
    'max_depth': [3],
    'n_estimators': [50],
    'learning_rate': [0.01]
}

# Initialize the XGBClassifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Initialize the GridSearchCV
grid_search = GridSearchCV(xgb, param_grid, cv=3, verbose=2)

# Fit the GridSearchCV
grid_search.fit(X_train_top_features, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train the XGBClassifier with the best parameters
xgb_best = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_best.fit(X_train_top_features, y_train)

# Make predictions on the training and test data
train_preds = xgb_best.predict(X_train_top_features)
test_preds = xgb_best.predict(X_test_top_features)

# Calculate and print the accuracy of the model on the training and test data
train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

# Print the classification report for the test data
report = classification_report(y_test, test_preds, target_names=le.classes_)

print("Best parameters:", best_params)
print("Training accuracy:", train_acc)
print("Test accuracy:", test_acc)
print("Classification report:\n", report)

# 保存模型
dump(xgb_best, 'model_.joblib')