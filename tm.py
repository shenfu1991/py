from flask import Flask, request, jsonify
import pandas as pd
from joblib import load

app = Flask(__name__)

# 加载模型和 LabelEncoder
model_tomo3m = load('model_tomo3m.joblib')
model_tomo5m = load('model_tomo5m.joblib')
model_tomo15m = load('model_tomo15m.joblib')
model_tomo30m = load('model_tomo30m.joblib')
model_tomo1h = load('model_tomo1h.joblib')
model_tomo4h = load('model_tomo4h.joblib')
le3m = load('label_encoder_tomo3m.joblib')
le5m = load('label_encoder_tomo5m.joblib')
le15m = load('label_encoder_tomo15m.joblib')
le30m = load('label_encoder_tomo30m.joblib')
le1h = load('label_encoder_tomo1h.joblib')
le4h = load('label_encoder_tomo4h.joblib')



@app.route('/predict_3m', methods=['POST'])
def predict_3m():
    return predict(model_tomo3m,le3m)

@app.route('/predict_5m', methods=['POST'])
def predict_5m():
    return predict(model_tomo5m,le5m)

@app.route('/predict_15m', methods=['POST'])
def predict_15m():
    return predict(model_tomo15m,le15m)

@app.route('/predict_30m', methods=['POST'])
def predict_30m():
    return predict(model_tomo30m,le30m)

@app.route('/predict_1h', methods=['POST'])
def predict_1h():
    return predict(model_tomo1h,le1h)

@app.route('/predict_4h', methods=['POST'])
def predict_4h():
    return predict(model_tomo4h,le4h)



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
