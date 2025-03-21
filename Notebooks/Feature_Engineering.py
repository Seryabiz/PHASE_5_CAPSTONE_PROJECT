# feature_engineering_pipeline.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

class ColumnNameFixer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'temparature' in X.columns:
            X.rename(columns={'temparature': 'temperature'}, inplace=True)
        return X

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, column, lower_quantile=0.05, upper_quantile=0.95):
        self.column = column
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        self.lower_bound = X[self.column].quantile(self.lower_quantile)
        self.upper_bound = X[self.column].quantile(self.upper_quantile)
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = np.clip(X[self.column], self.lower_bound, self.upper_bound)
        return X

class NewFeatureCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['temp_diff'] = X['maxtemp'] - X['mintemp']
        X['humidity_index'] = X['humidity'] / X['temperature']
        X['windspeed_category'] = pd.cut(X['windspeed'], bins=[0, 20, 40, np.inf], labels=['Low', 'Medium', 'High'])
        return X

def build_feature_engineering_pipeline(columns_to_cap):
    steps = [('fix_column_names', ColumnNameFixer())]
    for column in columns_to_cap:
        steps.append((f'cap_outliers_{column}', OutlierCapper(column=column)))
    steps.append(('create_new_features', NewFeatureCreator()))
    return Pipeline(steps=steps)

def plot_correlation_matrix(df):
    df_numeric = df.select_dtypes(include=['number'])
    corr_matrix = df_numeric.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.show()

def drop_highly_correlated_features(df, threshold=0.85, save_path='cleaned_train.csv'):
    corr_matrix = df.select_dtypes(include=['number']).corr()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr_features = [
        column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)
    ]
    df_dropped = df.drop(columns=high_corr_features)
    print("Removed Highly Correlated Features:", high_corr_features)
    df_dropped.to_csv(save_path, index=False)
    print(f"Cleaned training dataset saved as '{save_path}'")
    return df_dropped