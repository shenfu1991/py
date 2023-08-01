import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# 加载数据
file_path = 'ETHUSDT_3m.csv'
data = pd.read_csv(file_path)
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

# 重新采样到5分钟间隔
data.set_index('timestamp', inplace=True)
resampled_data = data['current'].resample('5T').mean().dropna().reset_index()

# 将数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(resampled_data['current'].values.reshape(-1, 1))

# 定义窗口大小
window_size = 5

# 构建LSTM所需的数据结构
X, y = [], []
for i in range(window_size, len(scaled_data) - 1):
    X.append(scaled_data[i - window_size:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 划分训练集和测试集
train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 构建LSTM模型
# 构建LSTM模型
model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))  # 更多的单元
model.add(LSTM(100, return_sequences=True))  # 添加更多层
model.add(LSTM(100))  # 添加更多层
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32)  # 更多的训练周期和更小的批次大小

# 其他代码保持不变


# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=64)

# 在测试集上进行预测
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# 计算均方误差
mse = mean_squared_error(y_test, predictions)

# 打印MSE
print('Mean Squared Error:', mse)

# 接下来，你可以使用相同的交易策略计算收益率
import matplotlib.pyplot as plt

# 绘制实际价格与预测价格的对比图
plt.figure(figsize=(14, 6))
plt.plot(y_test, label='Actual Prices')
plt.plot(predictions, label='Predicted Prices')
plt.title('BTC Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# 创建包含预测和实际价格的DataFrame
result_data = pd.DataFrame({'actual': y_test.flatten(), 'prediction': predictions.flatten()})
result_data['return'] = result_data['actual'].diff()
result_data['trade_signal'] = (result_data['prediction'] > result_data['actual'].shift(1)).astype(int) * 2 - 1
result_data['strategy_return'] = result_data['trade_signal'] * result_data['return']

# 计算策略的累积收益和收益率
initial_capital = 10000
leverage = 10
cumulative_return = result_data['strategy_return'].cumsum()
leveraged_return = cumulative_return.iloc[-1] * leverage
return_rate = leveraged_return / initial_capital * 100

print('Return Rate:', return_rate)

