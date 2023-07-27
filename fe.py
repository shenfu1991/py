import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
from openfe import OpenFE, transform

data = pd.read_csv('RDNTUSDT_15m_15m.csv')

sm = SMOTE(random_state=42)
X, y = sm.fit_resample(data.drop('result', axis=1), data['result'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Generate new features using OpenFE
ofe = OpenFE()
features = ofe.fit(data=X_train, label=y_train, n_jobs=1)  # generate new features
X_train, X_test = transform(X_train, X_test, features, n_jobs=1) # transform the train and test data according to generated features.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
print(classification_report(y_test, predictions))

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
