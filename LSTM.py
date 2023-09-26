import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Load the data
data = pd.read_csv("ALPHAUSDT_15m.csv")

# Create 'target' column
data['next_price'] = data['current'].shift(-1)
data['target'] = (data['next_price'] > data['current']).astype(int)
data.drop('next_price', axis=1, inplace=True)

counts = data['target'].value_counts()
print(counts)


# Data preprocessing
scaler = MinMaxScaler()
data['scaled_current'] = scaler.fit_transform(data[['current']])
train_size = int(len(data) * 0.8)
train, test = data.iloc[0:train_size], data.iloc[train_size:len(data)]

# Create dataset function
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# Create windowed dataset
TIME_STEPS = 150
X_train, y_train = create_dataset(train[['scaled_current']], train['target'], TIME_STEPS)
X_test, y_test = create_dataset(test[['scaled_current']], test['target'], TIME_STEPS)

# LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, shuffle=False)
