import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the data
data = pd.read_csv('cleaned_merged_15n.csv')  # replace with the path to your data

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

# Create a DataFrame for visualization
feature_importances = pd.DataFrame({"feature": X.columns, "importance": importances})

# Sort the DataFrame by importance
feature_importances = feature_importances.sort_values("importance", ascending=False)

# Print the feature importances
print(feature_importances)

# Select the top 3 features
top_features = feature_importances["feature"][:3]

print("Top 3 features:", top_features)


#       feature  importance
# 0     current    0.161568
# 1         avg    0.136167
# 9      signal    0.129838
# 7  volatility    0.122049
# 8       sharp    0.091632
# 3        high    0.075952
# 4         low    0.075866
# 2        open    0.074625
# 6      volume    0.066773
# 5        rate    0.065531