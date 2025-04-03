# preprocessing.py

import numpy as np
import pandas as pd
import os
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def save_pipeline(pipeline, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(pipeline, filename)


def load_pipeline(filename):
    return joblib.load(filename)


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column='day', period=31):
        self.column = column
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.column in X.columns:
            col_data = X[self.column]

            # Fix if it's a DataFrame (should be a Series)
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]

            col_data = pd.to_numeric(col_data, errors='coerce')

            if col_data.isnull().all():
                print(f"⚠️ All values in '{self.column}' are NaN after conversion.")
                return X

            X[f'{self.column}_sin'] = np.sin(2 * np.pi * col_data / self.period)
            X[f'{self.column}_cos'] = np.cos(2 * np.pi * col_data / self.period)
            X.drop(columns=[self.column], inplace=True)
        else:
            print(f"⚠️ Column '{self.column}' not found in dataset.")

        return X
    
class CyclicalFeaturesAdder(CyclicalEncoder):
    pass    




def build_preprocessing_pipeline(numeric_features, categorical_features):
    from Feature_Engineering import CyclicalFeaturesAdder

    steps = []

    # Only add cyclical if 'day' is in features
    if 'day' in numeric_features:
        steps.append(('cyclic', CyclicalFeaturesAdder(column='day', period=365)))
        numeric_features = numeric_features.drop('day')  # Remove 'day' so it's not processed again

    # Standard preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')

    steps.append(('preprocessor', preprocessor))

    return Pipeline(steps=steps)



