
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import classification_report

# ... (Rest of the original imports and setup code)

# Load the data
data_path = '/Users/xuanyuan/py/merged_ttt.csv'
data = pd.read_csv(data_path)

# Data Preprocessing: Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop('result', axis=1))
data_scaled = pd.DataFrame(scaled_features, columns=data.columns[:-1])

# Convert target labels to numerical values if they are not
encoder = LabelEncoder()
data['result'] = encoder.fit_transform(data['result'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data_scaled, data['result'], test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearch
param_grid = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'alpha': [0, 0.001, 0.01, 0.1],
    'lambda': [0, 0.001, 0.01, 0.1]
}
grid_clf = GridSearchCV(XGBClassifier(eval_metric='mlogloss'), param_grid, cv=5)
grid_clf.fit(X_train, y_train)
best_params = grid_clf.best_params_

print(best_params)

# Initialize and train XGBoost Classifier with best parameters and early stopping
clf = XGBClassifier(**best_params)
eval_set = [(X_train, y_train), (X_test, y_test)]
clf.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=10, verbose=True)

# Feature Importance Plot
plot_importance(clf)

# Model evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# ... (Rest of the original code for saving model etc.)
