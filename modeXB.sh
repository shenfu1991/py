#!/bin/bash  

arr="xgboost_model-k.pkl xgboost_model-k-r.pkl xgboost_model-k-dum.pkl xgboost_model-555.pkl xgboost_model-555-100.pkl xgboost_model-555-100.pkl"

ports=6601

fname=1

for name in $arr  
do  

yu="k"$fname".py"

cat>/root/$yu<<EOF

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
            modelName = "$name"
            with open(modelName, "rb") as pkl_file:
                loaded_data = pickle.load(pkl_file)
            cls._instance.model = loaded_data['model']
            cls._instance.scaler = loaded_data['scaler']
            cls._instance.label_encoder = loaded_data['label_encoder']
            cls._instance.feature_names = ['shortAvg','longAvg','volatility','diff']
        return cls._instance

#@app.route('/predict', methods=['POST'])
# Initialize resources outside of the predict function
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
    app.run(port=$ports,debug=True)


EOF

((fname = fname + 1))

((ports = ports + 1))

done
