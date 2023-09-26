import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

# 1. Load data
data = pd.read_csv('merged_5m-r.csv')

# 2. Data preprocessing
# Split data into features and target
X = data.drop('result', axis=1)
y = data['result']

# One-hot encode the target column
onehot_encoder = OneHotEncoder(sparse=False)
y_encoded = onehot_encoder.fit_transform(y.values.reshape(-1, 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data for LSTM (samples, timesteps, features)
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# 3. Build LSTM model
model = Sequential()
model.add(LSTM(5, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), activation='relu', return_sequences=True))
model.add(LSTM(5, activation='relu'))
model.add(Dense(4, activation='softmax'))  # 4 units for 4 categories

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train the model
model.fit(X_train_reshaped, y_train, epochs=2, batch_size=128, validation_data=(X_test_reshaped, y_test), verbose=1)

# 5. Evaluate the mod128
loss, accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
model.save('ks.h5')
