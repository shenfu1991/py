from flask import Flask, request, jsonify
import pickle
import numpy as np


app = Flask(__name__)

# Loading the models
model_3m = pickle.load(open('model__3m_3m.pkl', 'rb'))
model_5m = pickle.load(open('model__5m_5m.pkl', 'rb'))
model_15m = pickle.load(open('model__15m_15m.pkl', 'rb'))
model_30m = pickle.load(open('model__30m_30m.pkl', 'rb'))
model_1h = pickle.load(open('model__1h_1h.pkl', 'rb'))
model_4h = pickle.load(open('model__4h_4h.pkl', 'rb'))

# model_3m = load('model__3m_3m.pkl')
# model_5m = load('model__5m_5m.pkl')
# model_15m = load('model__15m_15m.pkl')
# model_30m = load('model__30m_30m.pkl')
# model_1h = load('model__1h_1h.pkl')
# model_4h = load('model__4h_4h.pkl')

@app.route('/predict_3m', methods=['POST'])
def predict_3m():
    return predict(model_3m)

@app.route('/predict_5m', methods=['POST'])
def predict_5m():
    return predict(model_5m)

@app.route('/predict_15m', methods=['POST'])
def predict_15m():
    return predict(model_15m)

@app.route('/predict_30m', methods=['POST'])
def predict_30m():
    return predict(model_30m)

@app.route('/predict_1h', methods=['POST'])
def predict_1h():
    return predict(model_1h)

@app.route('/predict_4h', methods=['POST'])
def predict_4h():
    return predict(model_4h)




def predict(model):
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
    app.run(port=5007, debug=True)
