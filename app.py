from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import xgboost

app = Flask(__name__)

# Load model, scaler and label encoder

modelName = "xgboost_model-l.pkl"
print(modelName)
print(xgboost.__version__)
feature_names = ['rank','upDownMa23','volatility','sharp','signal']
# feature_names = ['shortAvg','longAvg','volatility','diff']

with open(modelName, "rb") as pkl_file:
    loaded_data = pickle.load(pkl_file)
    model = loaded_data['model']
    scaler = loaded_data['scaler']
    label_encoder = loaded_data['label_encoder']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json['input']
        
        # Convert JSON data to DataFrame
        df = pd.DataFrame([data], columns=feature_names)
        
        # Scale the data
        scaled_data = scaler.transform(df)
        
        # Predict
        prediction = model.predict(scaled_data)
        
        # # Convert numeric prediction back to label
        # label_prediction = label_encoder.inverse_transform(prediction)

      
        if len(prediction.shape) > 1 and prediction.shape[1] > 1:
             prediction = np.argmax(prediction, axis=1)

        # Convert numeric prediction back to label
        label_prediction = label_encoder.inverse_transform(prediction)

        
        return jsonify([label_prediction[0]])
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(port=6600,debug=True)

