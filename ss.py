from flask import Flask, request, jsonify
import pandas as pd
from joblib import load

app = Flask(__name__)

# 加载模型和 LabelEncoder
model_3n = load('model_3n.joblib')
model_5n = load('model_5n.joblib')
model_15n = load('model_15n.joblib')
model_30n = load('model_30n.joblib')
model_1n = load('model_1n.joblib')
model_4n = load('model_4n.joblib')
le3n = load('label_encoder_3n.joblib')
le5n = load('label_encoder_5n.joblib')
le15n = load('label_encoder_15n.joblib')
le30n = load('label_encoder_30n.joblib')
le1n = load('label_encoder_1n.joblib')
le4n = load('label_encoder_4n.joblib')



@app.route('/predict_3m', methods=['POST'])
def predict_3m():
    return predict(model_3n,le3n)

@app.route('/predict_5m', methods=['POST'])
def predict_5m():
    return predict(model_5n,le5n)

@app.route('/predict_15m', methods=['POST'])
def predict_15m():
    return predict(model_15n,le15n)

@app.route('/predict_30m', methods=['POST'])
def predict_30m():
    return predict(model_30n,le30n)

@app.route('/predict_1h', methods=['POST'])
def predict_1h():
    return predict(model_1n,le1n)

@app.route('/predict_4h', methods=['POST'])
def predict_4h():
    return predict(model_4n,le4n)



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
    app.run(port=5001, debug=True)
