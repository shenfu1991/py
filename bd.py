import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek

df = pd.read_csv('merged_15mv3.csv')
X = df[['open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']]
y = df['result']

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

smote_tomek = SMOTETomek(random_state=1)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

model = RandomForestClassifier(class_weight='balanced')
model.fit(X_resampled, y_resampled) 

y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))