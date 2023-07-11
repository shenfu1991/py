from flask import Flask, request, jsonify
import pandas as pd
from joblib import load

app = Flask(__name__)

# 加载模型和 LabelEncoder
model_3d = load('model_3d.joblib')
model_5d = load('model_5d.joblib')
model_15d = load('model_15d.joblib')
model_30d = load('model_30d.joblib')
model_1d = load('model_1d.joblib')
model_4d = load('model_4d.joblib')
le3d = load('label_encoder_3d.joblib')
le5d = load('label_encoder_5d.joblib')
le15d = load('label_encoder_15d.joblib')
le30d = load('label_encoder_30d.joblib')
le1d = load('label_encoder_1d.joblib')
le4d = load('label_encoder_4d.joblib')



@app.route('/predict_3m', methods=['POST'])
def predict_3m():
    return predict(model_3d,le3d)

@app.route('/predict_5m', methods=['POST'])
def predict_5m():
    return predict(model_5d,le5d)

@app.route('/predict_15m', methods=['POST'])
def predict_15m():
    return predict(model_15d,le15d)

@app.route('/predict_30m', methods=['POST'])
def predict_30m():
    return predict(model_30d,le30d)

@app.route('/predict_1h', methods=['POST'])
def predict_1h():
    return predict(model_1d,le1d)

@app.route('/predict_4h', methods=['POST'])
def predict_4h():
    return predict(model_4d,le4d)



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
