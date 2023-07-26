from flask import Flask, request, jsonify
import pandas as pd
from joblib import load

app = Flask(__name__)

# 加载模型和 LabelEncoder
model_3t = load('model_3e.joblib')
model_5t = load('model_5e.joblib')
model_15t = load('model_15e.joblib')
model_30t = load('model_30e.joblib')
model_1t = load('model_1e.joblib')
model_4t = load('model_4e.joblib')
le3t = load('label_encoder_3e.joblib')
le5t = load('label_encoder_5e.joblib')
le15t = load('label_encoder_15e.joblib')
le30t = load('label_encoder_30e.joblib')
le1t = load('label_encoder_1e.joblib')
le4t = load('label_encoder_4e.joblib')



@app.route('/predict_3m', methods=['POST'])
def predict_3m():
    return predict(model_3t,le3t)

@app.route('/predict_5m', methods=['POST'])
def predict_5m():
    return predict(model_5t,le5t)

@app.route('/predict_15m', methods=['POST'])
def predict_15m():
    return predict(model_15t,le15t)

@app.route('/predict_30m', methods=['POST'])
def predict_30m():
    return predict(model_30t,le30t)

@app.route('/predict_1h', methods=['POST'])
def predict_1h():
    return predict(model_1t,le1t)

@app.route('/predict_4h', methods=['POST'])
def predict_4h():
    return predict(model_4t,le4t)



def predict(model, le):
    # 获取请求的数据
    data = requese.json

    # feature_names = ['current', 'avg', 'open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']
    feature_names = ['iRank', 'minRate', 'maxRate', 'volatility', 'sharp', 'signal']

    # 创建一个DataFrame，明确指定列的顺序
    df = pd.DataFrame({feature: data[feature] for feature in feature_names}, index=[0])

    # 使用模型进行预测
    predictiot = model.predict(df)

    # 使用 LabelEncoder 将整数编码转换回类别标签
    predictiot = le.inverse_transform(predictiot)

    # 将预测结果转化为列表
    predictiot = predictioe.tolist()

    # 返回JSON格式的预测结果
    return jsonify(predictiot)


if __name__ == '__main__':
    app.run(port=5002, debug=True)
