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
model = load('model_.joblib')
preprocessing = load('preprocessing.joblib')

@app.route('/predict', methods=['POST'])
def predict():
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
    app.run(port=5001, debug=True)
