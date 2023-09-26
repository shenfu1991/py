from flask import Flask, request, jsonify
import pickle
import os
import psutil
import pandas as pd

app = Flask(__name__)

class SingletonModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("Creating Singleton Instance")
            cls._instance = super(SingletonModel, cls).__new__(cls)
            modelName = "xgboost_model-k.pkl"
            with open(modelName, "rb") as pkl_file:
                loaded_data = pickle.load(pkl_file)
            cls._instance.model = loaded_data['model']
            cls._instance.scaler = loaded_data['scaler']
            cls._instance.label_encoder = loaded_data['label_encoder']
            cls._instance.feature_names = ['shortAvg','longAvg','volatility','diff']
        return cls._instance

resources = SingletonModel()
model = resources.model
scaler = resources.scaler
label_encoder = resources.label_encoder


@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler, label_encoder

    # Get JSON data from request
    data = request.json['input']

    # Convert JSON data to DataFrame
    df = pd.DataFrame([data], columns=resources.feature_names)

    # Scale the data
    scaled_data = scaler.transform(df)

    # Predict
    prediction = model.predict(scaled_data)

    # Convert numeric prediction back to label
    label_prediction = label_encoder.inverse_transform(prediction)

    return jsonify([label_prediction[0]])

if __name__ == '__main__':
    app.run(port=6601,debug=True)

