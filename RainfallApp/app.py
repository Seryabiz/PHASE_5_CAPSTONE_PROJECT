import sys
import os
sys.path.append(os.path.abspath("../Notebooks"))

import streamlit as st
import pandas as pd
import joblib


st.set_page_config(page_title="Rainfall Prediction", layout="centered")

# Load model and pipeline
model = joblib.load('./Models/best_stacking_ensemble_tuned.pkl')
pipeline = joblib.load('./Models/full_preprocessing_pipeline.joblib')
df_refined = pd.read_csv('./Data/refined_train.csv')

st.title("🌧️ Rainfall Prediction App")
st.write("Upload a pre-engineered Excel file to get rainfall predictions.")

uploaded_file = st.file_uploader("Upload your .xlsx file", type=["xlsx"])

if uploaded_file is not None:
    try:
        new_data = pd.read_excel(uploaded_file)
        st.write("✅ Data Preview", new_data.head())

        selected_columns = [col for col in df_refined.columns if col != 'rainfall']
        new_data_selected = new_data[selected_columns]

        new_data_processed = pipeline.transform(new_data_selected)
        prediction = model.predict_proba(new_data_processed)[:, 1]

        st.success(f"🌦️ Predicted Rainfall Probability: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
