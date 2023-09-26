import asyncio
from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# 模型相关全局变量
model = None
scaler = None 
label_encoder = None

async def load_model():

  global model, scaler, label_encoder,feature_names

  if not model:

    with open('xgboost_model.pkl', 'rb') as f:
      loaded_data = pickle.load(f)
      model = loaded_data['model']
      scaler = loaded_data['scaler'] 
      label_encoder = loaded_data['label_encoder']
      # feature_names = ['shortAvg','longAvg','volatility','diff']
      feature_names = ['rank','minR','maxR','minDiffR','maxDiffR','topRank']

async def predict(data):

  await load_model()

    # Convert JSON data to DataFrame
  df = pd.DataFrame([data], columns=feature_names)

    # Scale the data
  scaled_data = scaler.transform(df)

    # Predict
  prediction = model.predict(scaled_data)

    # Convert numeric prediction back to label
  label_prediction = label_encoder.inverse_transform(prediction)

  return [label_prediction[0]]


@app.route('/predict', methods=['POST'])
async def predict_handler():

  data = request.json['input']
  result = await asyncio.gather(predict(data))

  return jsonify(result)

if __name__ == '__main__':
    app.run(port=6601,debug=True)
