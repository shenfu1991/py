from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model, label encoder from the pickle file
with open("xgboost_model_xgb.pkl", "rb") as pkl_file:
    data = pickle.load(pkl_file)
    model = data['model']
    label_encoder = data['label_encoder']

# Define the feature names as they were during training
feature_names = ['iRank', 'minRate', 'maxRate', 'volatility', 'sharp', 'signal', 'minR', 'maxR']

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Log that the endpoint was hit
        print("Received request at /predict endpoint")
        
        # Get the request data
        data = request.json
        
        # Check if "input" key is present in the data
        if "input" not in data:
            return jsonify({"error": "Key 'input' not found in the request data."}), 400

        # Convert data to dataframe with correct feature names
        df = pd.DataFrame([data['input']], columns=feature_names)
        
        # Make prediction
        prediction = model.predict(df)
        decoded_prediction = label_encoder.inverse_transform(prediction)
        
        # Return the prediction
        return jsonify([decoded_prediction[0]])
    
    except Exception as e:
        # Log the error
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

# The rest of the Flask application remains the same



# This is just for demonstration. When deploying to production, use a production-ready server like Gunicorn.
if __name__ == "__main__":
    app.run(port=6600,debug=True)
