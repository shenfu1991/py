from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model, scaler and label encoder
with open("xgboost_model.pkl", "rb") as pkl_file:
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
        df = pd.DataFrame([data])
        
        # Scale the data
        scaled_data = scaler.transform(df)
        
        # Predict
        prediction = model.predict(scaled_data)
        
        # Convert numeric prediction back to label
        label_prediction = label_encoder.inverse_transform(prediction)
        
        return jsonify([label_prediction[0]])
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
