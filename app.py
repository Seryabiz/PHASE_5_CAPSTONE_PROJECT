from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def Home():
    return "Welcome to the rainfall prediction site"

@app.route('/predict', methods = ['POST'])
def Prediction():
    model = joblib.load('model.pkl')
    data = request.json
    prediction = model.predict_proba()[1]
    return prediction