from flask import Flask, request, jsonify
import pandas as pd
from joblib import load

app = Flask(__name__)

# 加载模型和 LabelEncoder
model_3r = load('model_3r.joblib')
model_5r = load('model_5r.joblib')
model_15r = load('model_15r.joblib')
model_30r = load('model_30r.joblib')
model_1r = load('model_1r.joblib')
model_4r = load('model_4r.joblib')
le3r = load('label_encoder_3r.joblib')
le5r = load('label_encoder_5r.joblib')
le15r = load('label_encoder_15r.joblib')
le30r = load('label_encoder_30r.joblib')
le1r = load('label_encoder_1r.joblib')
le4r = load('label_encoder_4r.joblib')



@app.route('/predict_3m', methods=['POST'])
def predict_3m():
    return predict(model_3r,le3r)

@app.route('/predict_5m', methods=['POST'])
def predict_5m():
    return predict(model_5r,le5r)

@app.route('/predict_15m', methods=['POST'])
def predict_15m():
    return predict(model_15r,le15r)

@app.route('/predict_30m', methods=['POST'])
def predict_30m():
    return predict(model_30r,le30r)

@app.route('/predict_1h', methods=['POST'])
def predict_1h():
    return predict(model_1r,le1r)

@app.route('/predict_4h', methods=['POST'])
def predict_4h():
    return predict(model_4r,le4r)



def predict(model, le):
    # 获取请求的数据
    data = request.json

    # feature_names = ['current', 'avg', 'open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']
    feature_names = ['iRank', 'minRate', 'maxRate', 'volatility', 'sharp', 'signal']

    # 创建一个DataFrame，明确指定列的顺序
    df = pd.DataFrame({feature: data[feature] for feature in feature_names}, index=[0])

    # 使用模型进行预测
    predictior = model.predict(df)

    # 使用 LabelEncoder 将整数编码转换回类别标签
    predictior = le.inverse_transform(predictior)

    # 将预测结果转化为列表
    predictior = predictior.tolist()

    # 返回JSON格式的预测结果
    return jsonify(predictior)


if __name__ == '__main__':
    app.run(port=5001, debug=True)
