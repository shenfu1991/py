from flask import Flask, request, jsonify
import pandas as pd
from joblib import load

app = Flask(__name__)

# 加载模型和 LabelEncoder
model_3m = load('model_3m.joblib')
model_5m = load('model_5m.joblib')
model_15m = load('model_15m.joblib')
model_30m = load('model_30m.joblib')
le3m = load('label_encoder_3m.joblib')
le5m = load('label_encoder_5m.joblib')
le15m = load('label_encoder_15m.joblib')
le30m = load('label_encoder_30m.joblib')

model_3mv2 = load('model_3mv2.joblib')
model_5mv2 = load('model_5mv2.joblib')
model_15mv2 = load('model_15mv2.joblib')
model_30mv2 = load('model_30mv2.joblib')
le3mv2 = load('label_encoder_3mv2.joblib')
le5mv2 = load('label_encoder_5mv2.joblib')
le15mv2 = load('label_encoder_15mv2.joblib')
le30mv2 = load('label_encoder_30mv2.joblib')





@app.route('/predict_3m', methods=['POST'])
def predict_3m():
    return predict(model_3m,le3m)

@app.route('/predict_5m', methods=['POST'])
def predict_5m():
    return predict(model_5m,le5m)

@app.route('/predict_15m', methods=['POST'])
def predict_15m():
    return predict(model_15m,le15m)

@app.route('/predict_30m', methods=['POST'])
def predict_30m():
    return predict(model_30m,le30m)


@app.route('/predict_3mv2', methods=['POST'])
def predict_3mv2():
    return predict(model_3mv2,le3mv2)

@app.route('/predict_5mv2', methods=['POST'])
def predict_5mv2():
    return predict(model_5mv2,le5mv2)

@app.route('/predict_15mv2', methods=['POST'])
def predict_15mv2():
    return predict(model_15mv2,le15mv2)

@app.route('/predict_30mv2', methods=['POST'])
def predict_30mv2():
    return predict(model_30mv2,le30mv2)



def predict(model,le):
    # 获取请求的数据
    data = request.json

    # 以相同的顺序指定特征名称
    feature_names = ['open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']

    # 创建一个DataFrame，明确指定列的顺序
    df = pd.DataFrame(data, columns=feature_names, index=[0])

    # 使用模型进行预测
    prediction = model.predict(df)

    # 使用 LabelEncoder 将整数编码转换回类别标签
    prediction = le.inverse_transform(prediction)

    # 将预测结果转化为列表
    prediction = prediction.tolist()

    # 返回JSON格式的预测结果
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
