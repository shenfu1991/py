from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the model
with open('model__15m_15m.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)

    # Convert the data into numpy array
    predict_request = np.array(data['input'])

    # Use the model to predict
    prediction = model.predict(predict_request.reshape(1, -1))

    # Convert numpy array to list
    output = prediction.tolist()

    # Return the result
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
