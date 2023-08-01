from flask import Flask, request, jsonify
import pandas as pd
from joblib import load

app = Flask(__name__)

# 加载模型和 LabelEncoder
model_3r = load('btc_price_predictor.joblib')


@app.route('/predict', methods=['POST'])
def predict_3m():
    return predict(model_3r)


def predict(model):
    # 获取请求的数据
    data = request.json

    feature_names = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'moving_avg_5','moving_avg_10']

    # 创建一个DataFrame，明确指定列的顺序
    df = pd.DataFrame({feature: data[feature] for feature in feature_names}, index=[0])

    # 使用模型进行预测
    predictior = model.predict(df)

    # # 将预测结果转化为列表
    predictior = predictior.tolist()

    # 返回JSON格式的预测结果
    return jsonify(predictior)


if __name__ == '__main__':
    app.run(port=5005, debug=True)
