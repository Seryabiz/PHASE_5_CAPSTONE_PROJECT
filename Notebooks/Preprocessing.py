# preprocessing.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

class CyclicalFeaturesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, column='day', period=365):
        self.column = column
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[f'{self.column}_sin'] = np.sin(2 * np.pi * X[self.column] / self.period)
        X[f'{self.column}_cos'] = np.cos(2 * np.pi * X[self.column] / self.period)
        X.drop(columns=[self.column], inplace=True)
        return X

class TempRangeAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['temp_range'] = X['maxtemp'] - X['mintemp']
        return X

def build_preprocessing_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    feature_engineering = Pipeline(steps=[
        ('cyclic', CyclicalFeaturesAdder()),
        ('temp_range', TempRangeAdder())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')

    full_pipeline = Pipeline(steps=[
        ('feature_engineering', feature_engineering),
        ('preprocessor', preprocessor)
    ])

    return full_pipeline

def save_pipeline(pipeline, filename):
    joblib.dump(pipeline, filename)

def load_pipeline(filename):
    return joblib.load(filename)
