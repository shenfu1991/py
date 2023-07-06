from flask import Flask, request, jsonify
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from joblib import dump


app = Flask(__name__)


class PreprocessingPipeline:
    def __init__(self, scaler, rfe, pca, le):
        self.scaler = scaler
        self.rfe = rfe
        self.pca = pca
        self.le = le


# 加载模型和预处理流水线
model_3mv9 = load('model_3mv9.joblib')
preprocessing_3mv9 = load('preprocessing_3mv9.joblib')

model_30mv9 = load('model_30mv9.joblib')
preprocessing_30mv9 = load('preprocessing_30mv9.joblib')

model_5mv9 = load('model_5mv9.joblib')
preprocessing_5mv9 = load('preprocessing_5mv9.joblib')

model_15mv9 = load('model_15mv9.joblib')
preprocessing_15mv9 = load('preprocessing_15mv9.joblib')

model_1hv9 = load('model_1hv9.joblib')
preprocessing_1hv9 = load('preprocessing_1hv9.joblib')

model_4hv9 = load('model_4hv9.joblib')
preprocessing_4hv9 = load('preprocessing_4hv9.joblib')

@app.route('/predict_3m', methods=['POST'])
def predict_3m():
    return predict(model_3mv9,preprocessing_3mv9)

@app.route('/predict_30m', methods=['POST'])
def predict_30m():
    return predict(model_30mv9,preprocessing_30mv9)

@app.route('/predict_5m', methods=['POST'])
def predict_5m():
    return predict(model_5mv9,preprocessing_5mv9)

@app.route('/predict_15m', methods=['POST'])
def predict_15m():
    return predict(model_15mv9,preprocessing_15mv9)

@app.route('/predict_1h', methods=['POST'])
def predict_1h():
    return predict(model_1hv9,preprocessing_1hv9)

@app.route('/predict_4h', methods=['POST'])
def predict_4h():
    return predict(model_4hv9,preprocessing_4hv9)





def predict(model,preprocessing):
    # 获取请求的数据
    data = request.json

    feature_names = ['current', 'avg', 'open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']

    # 创建一个DataFrame，明确指定列的顺序
    df = pd.DataFrame({feature: data[feature] for feature in feature_names}, index=[0])

    # 标准化处理
    X = preprocessing.scaler.transform(df)

    # 递归特征消除处理
    X = preprocessing.rfe.transform(X)

    # 主成分分析处理
    X = preprocessing.pca.transform(X)

    # 使用模型进行预测
    prediction = model.predict(X)

    # 使用 LabelEncoder 将整数编码转换回类别标签
    prediction = preprocessing.le.inverse_transform(prediction)

    # 将预测结果转化为列表
    prediction = prediction.tolist()

    # 返回JSON格式的预测结果
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(port=5002, debug=True)
