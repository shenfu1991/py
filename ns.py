from flask import Flask, request, jsonify
import pandas as pd
from joblib import load

app = Flask(__name__)

# 加载模型和 LabelEncoder
model_3v4 = load('model_3v4.joblib')
model_5v4 = load('model_5v4.joblib')
model_15v4 = load('model_15v4.joblib')
model_30v4 = load('model_30v4.joblib')
model_1v4 = load('model_1v4.joblib')
model_4v4 = load('model_4v4.joblib')
le3v4 = load('label_encoder_3v4.joblib')
le5v4 = load('label_encoder_5v4.joblib')
le15v4 = load('label_encoder_15v4.joblib')
le30v4 = load('label_encoder_30v4.joblib')
le1v4 = load('label_encoder_1v4.joblib')
le4v4 = load('label_encoder_4v4.joblib')



@app.route('/predict_3m', methods=['POST'])
def predict_3m():
    return predict(model_3v4,le3v4)

@app.route('/predict_5m', methods=['POST'])
def predict_5m():
    return predict(model_5v4,le5v4)

@app.route('/predict_15m', methods=['POST'])
def predict_15m():
    return predict(model_15v4,le15v4)

@app.route('/predict_30m', methods=['POST'])
def predict_30m():
    return predict(model_30v4,le30v4)

@app.route('/predict_1h', methods=['POST'])
def predict_1h():
    return predict(model_1v4,le1v4)

@app.route('/predict_4h', methods=['POST'])
def predict_4h():
    return predict(model_4v4,le4v4)



def predict(model, le):
    # 获取请求的数据
    data = request.json

    feature_names = ['current', 'avg', 'open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']

    # 创建一个DataFrame，明确指定列的顺序
    df = pd.DataFrame({feature: data[feature] for feature in feature_names}, index=[0])

    # 使用模型进行预测
    prediction = model.predict(df)

    # 使用 LabelEncoder 将整数编码转换回类别标签
    prediction = le.inverse_transform(prediction)

    # 将预测结果转化为列表
    prediction = prediction.tolist()

    # 返回JSON格式的预测结果
    return jsonify(prediction)


if __name__ == '__main__':
    app.run(port=5002, debug=True)
