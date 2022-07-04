# app.py

from flask import Flask, request, jsonify
from joblib import load, dump

rfc = load("models/rfc.joblib")
scaler = load("models/scaler.joblib")

app = Flask(__name__)

@app.route('/')
def serve():
    return jsonify(success=True)

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.get_json()
    print(data)
    scaled_data = scaler.transform([data["data"]])
    print(scaled_data)
    prediction = rfc.predict(scaled_data)
    print(prediction)
    response = {
        "result":int(prediction[0])
    }
    return jsonify(response)

# We only need this for local development.
if __name__ == '__main__':
    app.run()
