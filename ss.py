from flask import Flask, request, jsonify
import pandas as pd
from joblib import load

app = Flask(__name__)

# 加载模型和 LabelEncoder
model_3mv3 = load('model_3mv3.joblib')
model_5mv3 = load('model_5mv3.joblib')
model_15mv3 = load('model_15mv3.joblib')
model_30mv3 = load('model_30mv3.joblib')
model_1hv3 = load('model_1hv3.joblib')
model_4hv3 = load('model_4hv3.joblib')
le3mv3 = load('label_encoder_3mv3.joblib')
le5mv3 = load('label_encoder_5mv3.joblib')
le15mv3 = load('label_encoder_15mv3.joblib')
le30mv3 = load('label_encoder_30mv3.joblib')
le1hv3 = load('label_encoder_1hv3.joblib')
le4hv3 = load('label_encoder_4hv3.joblib')

model_3mv4 = load('model_3mv4.joblib')
model_5mv4 = load('model_5mv4.joblib')
model_15mv4 = load('model_15mv4.joblib')
model_30mv4 = load('model_30mv4.joblib')
model_1hv4 = load('model_1hv3.joblib')
model_4hv4 = load('model_4hv3.joblib')
le3mv4 = load('label_encoder_3mv4.joblib')
le5mv4 = load('label_encoder_5mv4.joblib')
le15mv4 = load('label_encoder_15mv4.joblib')
le30mv4 = load('label_encoder_30mv4.joblib')
le1hv4 = load('label_encoder_1hv4.joblib')
le4hv4 = load('label_encoder_4hv4.joblib')





@app.route('/predict_3m', methods=['POST'])
def predict_3m():
    return predict(model_3mv3,le3mv3)

@app.route('/predict_5m', methods=['POST'])
def predict_5m():
    return predict(model_5mv3,le5mv3)

@app.route('/predict_15m', methods=['POST'])
def predict_15m():
    return predict(model_15mv3,le15mv3)

@app.route('/predict_30m', methods=['POST'])
def predict_30m():
    return predict(model_30mv3,le30mv3)

@app.route('/predict_1h', methods=['POST'])
def predict_1h():
    return predict(model_1hv3,le1hv3)

@app.route('/predict_4h', methods=['POST'])
def predict_4h():
    return predict(model_4hv3,le4hv3)


@app.route('/predict_3mv4', methods=['POST'])
def predict_3mv4():
    return predict(model_3mv4,le3mv4)

@app.route('/predict_5mv4', methods=['POST'])
def predict_5mv4():
    return predict(model_5mv4,le5mv4)

@app.route('/predict_15mv4', methods=['POST'])
def predict_15mv4():
    return predict(model_15mv4,le15mv4)

@app.route('/predict_30mv4', methods=['POST'])
def predict_30mv4():
    return predict(model_30mv4,le30mv4)

@app.route('/predict_1hv4', methods=['POST'])
def predict_1hv4():
    return predict(model_1hv4,le1hv4)

@app.route('/predict_4hv4', methods=['POST'])
def predict_4hv4():
    return predict(model_4hv4,le4hv4)




def predict(model,le):
    # 获取请求的数据
    data = request.json

    # 以相同的顺序指定特征名称
    feature_names = ['open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']

    # 创建一个DataFrame，明确指定列的顺序
    df = pd.DataFrame(data, columns=feature_names, index=[0])

    print(df)

    # 使用模型进行预测
    prediction = model.predict(df)

    # 使用 LabelEncoder 将整数编码转换回类别标签
    prediction = le.inverse_transform(prediction)

    print(prediction)
    # 将预测结果转化为列表
    prediction = prediction.tolist()

    print(prediction)

    # 返回JSON格式的预测结果
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
