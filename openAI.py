import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
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

# Select the top 3 features
top_features = ["current", "signal", "avg"] # replace this with your top features
X_train_top_features = X_train[top_features]

# Define the parameter grid
param_grid = {
    'max_depth': [7],
    'n_estimators': [150],
    'learning_rate': [0.2]
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

# Save the model
dump(xgb_best, 'model_.joblib')

# Save the label encoder
dump(le, 'label_encoder.joblib')
