import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from joblib import dump

# Load the data
data = pd.read_csv('cleaned_merged_15n.csv')

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
#top_features = ['current','avg','open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']

X_train_top_features = X_train[top_features]

# Define the parameter grid
param_grid = {
    'max_depth': [3,7,10,15],
    'n_estimators': [10,100,150,200],
    'learning_rate': [0.001,0.01,0.1,0.2]
}

# Initialize the XGBClassifier
xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)

# Initialize the GridSearchCV
grid_search = GridSearchCV(xgb, param_grid, cv=3, verbose=2)

# Fit the GridSearchCV
grid_search.fit(X_train_top_features, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train the XGBClassifier with the best parameters
xgb_best = XGBClassifier(**best_params, eval_metric='mlogloss', random_state=42)
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
