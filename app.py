from flask import Flask, request, jsonify
import joblib
import sys
import pandas as pd
sys.path.append('./Notebooks')
import Feature_Engineering

feature_pipeline = joblib.load('feature_engineering.pkl')

columns = joblib.load('columns.pkl')

pipeline = joblib.load('Preprocessing_pipeline.joblib')

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def Home():
    return "Welcome to the rainfall prediction site"

@app.route('/predict', methods = ['POST'])
def Prediction():
    model = joblib.load('model.pkl')
    data = pd.DataFrame([request.json])
    features = feature_pipeline.transform(data)
    selected_features = features[[x for x in columns if x != 'rainfall']]
    X = pipeline.transform(selected_features)
    prediction = model.predict_proba(X)
    return jsonify({'Probability of rain':prediction.tolist()[0][1]})