from flask import Flask, request, render_template
import joblib
import sys
import pandas as pd
sys.path.append('./Notebooks')
import Feature_Engineering
import Preprocessing

feature_pipeline = joblib.load('feature_engineering.pkl')

columns = joblib.load('columns.pkl')

pipeline = joblib.load('Preprocessing_pipeline.pkl')

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def Home():
    model = joblib.load('model.pkl')
    prediction = None
    if request.method == 'POST':
        # Extract data from the form
        values = [
            float(request.form['id']),
            float(request.form['day']),
            float(request.form['pressure']),
            float(request.form['maxtemp']),
            float(request.form['temparature']),
            float(request.form['mintemp']),
            float(request.form['dewpoint']),
            float(request.form['humidity']),
            float(request.form['cloud']),
            float(request.form['sunshine']),
            float(request.form['winddirection']),
            float(request.form['windspeed'])
        ]
        data = pd.DataFrame([values],columns=['id','day','pressure','maxtemp',
                                            'temparature','mintemp','dewpoint','humidity',
                                            'cloud','sunshine','winddirection',
                                            'windspeed'])
        features = feature_pipeline.transform(data)
        selected_features = features[[x for x in columns if x != 'rainfall']]
        X = pipeline.transform(selected_features)
        prediction = model.predict_proba(X)
        prediction = prediction[0][1] 
    return render_template('index.html',prediction=prediction)
if __name__=='__main__':
    app.run(debug=True)