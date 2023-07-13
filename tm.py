from flask import Flask, request, jsonify
import pandas as pd
from joblib import load

app = Flask(__name__)

# 加载模型和 LabelEncoder
model_3b = load('model_3b.joblib')
model_5b = load('model_5b.joblib')
model_15b = load('model_15b.joblib')
model_30b = load('model_30b.joblib')
model_1b = load('model_1b.joblib')
model_4b = load('model_4b.joblib')
le3b = load('label_encoder_3b.joblib')
le5b = load('label_encoder_5b.joblib')
le15b = load('label_encoder_15b.joblib')
le30b = load('label_encoder_30b.joblib')
le1b = load('label_encoder_1b.joblib')
le4b = load('label_encoder_4b.joblib')



@app.route('/predict_3m', methods=['POST'])
def predict_3m():
    return predict(model_3b,le3b)

@app.route('/predict_5m', methods=['POST'])
def predict_5m():
    return predict(model_5b,le5b)

@app.route('/predict_15m', methods=['POST'])
def predict_15m():
    return predict(model_15b,le15b)

@app.route('/predict_30m', methods=['POST'])
def predict_30m():
    return predict(model_30b,le30b)

@app.route('/predict_1h', methods=['POST'])
def predict_1h():
    return predict(model_1b,le1b)

@app.route('/predict_4h', methods=['POST'])
def predict_4h():
    return predict(model_4b,le4b)



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
    app.run(port=5003, debug=True)
