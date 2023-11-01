from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import xgboost

app = Flask(__name__)



with open("xgboost_model-5m-dum.pkl", "rb") as pkl_file:
    loaded_data1 = pickle.load(pkl_file)
    model1 = loaded_data1['model']
    scaler1 = loaded_data1['scaler']
    label_encoder1 = loaded_data1['label_encoder']


with open("xgboost_model-dum.pkl", "rb") as pkl_file:
    loaded_data2 = pickle.load(pkl_file)
    model2 = loaded_data2['model']
    scaler2 = loaded_data2['scaler']
    label_encoder2 = loaded_data2['label_encoder']

 with open("xgboost_model-30m-dum.pkl", "rb") as pkl_file:
     loaded_data3 = pickle.load(pkl_file)
     model3 = loaded_data3['model']
     scaler3 = loaded_data3['scaler']
     label_encoder3 = loaded_data3['label_encoder']


@app.route('/predict_3m', methods=['POST'])
def predict_3m():
    return predict(model1,scaler1,label_encoder1)

@app.route('/predict_5m', methods=['POST'])
def predict_5m():
    return predict(model1,scaler1,label_encoder1)

@app.route('/predict_15m', methods=['POST'])
def predict_15m():
    return predict(model2,scaler2,label_encoder2)

@app.route('/predict_30m', methods=['POST'])
def predict_30m():
    return predict(model3,scaler3,label_encoder3)

@app.route('/predict_1h', methods=['POST'])
def predict_1h():
    return predict(model3,scaler3,label_encoder3)

@app.route('/predict_4h', methods=['POST'])
def predict_4h():
    return predict(model3,scaler3,label_encoder3)





def predict(model, scaler,label_encoder): 
    try:
        # Get JSON data from request
        data = request.json['input']
        
        # Convert JSON data to DataFrame
        df = pd.DataFrame([data])
        
        # Scale the data
        scaled_data = scaler.transform(df)
        
        # Predict
        prediction = model.predict(scaled_data)
      
        if len(prediction.shape) > 1 and prediction.shape[1] > 1:
             prediction = np.argmax(prediction, axis=1)

        # Convert numeric prediction back to label
        label_prediction = label_encoder.inverse_transform(prediction)
        
        return jsonify([label_prediction[0]])
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    #app.run()
    app.run(port=6601,debug=True)

