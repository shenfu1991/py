from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Loading the models
model_3m = pickle.load(open('model_3m.pkl', 'rb'))
model_5m = pickle.load(open('model_5m.pkl', 'rb'))
model_15m = pickle.load(open('model_15m.pkl', 'rb'))
model_30m = pickle.load(open('model_30m.pkl', 'rb'))
model_1h = pickle.load(open('model_1h.pkl', 'rb'))
model_4h = pickle.load(open('model_4h.pkl', 'rb'))

def predict(model, data):
    # Preprocessing steps if needed
    # data_scaled = scaler.transform(data)
    return model.predict(data)

@app.route('/predict_3m', methods=['POST'])
def predict_3m():
    data = request.json
    prediction = predict(model_3m, data)
    return jsonify(prediction.tolist())

@app.route('/predict_5m', methods=['POST'])
def predict_5m():
    data = request.json
    prediction = predict(model_5m, data)
    return jsonify(prediction.tolist())

@app.route('/predict_15m', methods=['POST'])
def predict_15m():
    data = request.json
    prediction = predict(model_15m, data)
    return jsonify(prediction.tolist())

@app.route('/predict_30m', methods=['POST'])
def predict_30m():
    data = request.json
    prediction = predict(model_30m, data)
    return jsonify(prediction.tolist())

@app.route('/predict_1h', methods=['POST'])
def predict_1h():
    data = request.json
    prediction = predict(model_1h, data)
    return jsonify(prediction.tolist())

@app.route('/predict_4h', methods=['POST'])
def predict_4h():
    data = request.json
    prediction = predict(model_4h, data)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run()
