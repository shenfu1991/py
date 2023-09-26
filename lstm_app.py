from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import OneHotEncoder


app = Flask(__name__)



# Load the model
model = load_model('ks.h5')
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json(force=True)
    
    # Convert data into numpy array
    features = np.array([data['input']])

    # Define and fit onehot_encoder
    onehot_encoder = OneHotEncoder(sparse=False)
    y_reshaped = data['result'].values.reshape(-1, 1)
    onehot_encoder.fit(y_reshaped)
    
    # Reshape the data for LSTM (samples, timesteps, features)
    reshaped_features = features.reshape(features.shape[0], 1, features.shape[1])
    
    # Make prediction
    prediction = model.predict(reshaped_features)
    
    # Convert prediction to label
    predicted_onehot = np.zeros_like(prediction)
    predicted_onehot[0][np.argmax(prediction)] = 1
    predicted_label = onehot_encoder.inverse_transform(predicted_onehot)[0][0]
    
    
    # Return the predicted label
    return jsonify([predicted_label])

if __name__ == '__main__':
    app.run(port=6600, debug=True)
