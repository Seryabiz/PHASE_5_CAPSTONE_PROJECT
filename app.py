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
    return"""
    <!DOCTYPE html>
    <html>
    <head>
    <!--metadata about this document-->
    </head>
    <body>
    <!--this is the information to be displayed-->
    <h1>Welcome to the rainfall prediction website</h1>
    <p>
    <h3>Features used in the prediction</h3>
    <ul>
    <li>id</li>
    <li>day</li>
    <li>pressure</li>
    <li>maxtemp</li>
    <li>temparature</li>
    <li>mintemp</li>
    <li>dewpoint</li>
    <li>humidity</li>
    <li>cloud</li>
    <li>sunshine</li>
    <li>winddirection</li>
    <li>windspeed</li>
    </ul>
    </p>
    </body>
    </html>
    """
@app.route('/predict', methods = ['POST'])
def Prediction():
    model = joblib.load('model.pkl')
    data = pd.DataFrame([request.json])
    features = feature_pipeline.transform(data)
    selected_features = features[[x for x in columns if x != 'rainfall']]
    X = pipeline.transform(selected_features)
    prediction = model.predict_proba(X)
    return jsonify({'Probability of rain':prediction.tolist()[0][1]})