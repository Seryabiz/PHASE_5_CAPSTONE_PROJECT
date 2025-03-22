# feature_engineering_pipeline.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

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
        self.upper_quantile = upper_quantile  # <- Add this line

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
        X['humidity_index'] = X['humidity'] / (X['temperature'] + 1e-6)
        X['windspeed_category'] = pd.cut(X['windspeed'], bins=[0, 20, 40, np.inf], labels=['Low', 'Medium', 'High'])
        X['temp_sum'] = X['maxtemp'] + X['mintemp']
        X['dewpoint_humidity_ratio'] = X['dewpoint'] / (X['humidity'] + 1e-6)
        return X

class PolynomialFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2, include_bias=False):
        self.degree = degree
        self.include_bias = include_bias
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=self.include_bias)
        self.feature_names = None

    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        self.poly.fit(X[numeric_cols])
        self.feature_names = self.poly.get_feature_names_out(numeric_cols)
        return self

    def transform(self, X):
        X = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        poly_features = self.poly.transform(X[numeric_cols])
        poly_df = pd.DataFrame(poly_features, columns=self.feature_names, index=X.index)
        X = pd.concat([X, poly_df.drop(columns=numeric_cols)], axis=1)  # Avoid duplicating existing features
        return X

def build_feature_engineering_pipeline(columns_to_cap):
    steps = [('fix_column_names', ColumnNameFixer())]
    for column in columns_to_cap:
        steps.append((f'cap_outliers_{column}', OutlierCapper(column=column)))
    steps.append(('create_new_features', NewFeatureCreator()))
    steps.append(('add_polynomial_features', PolynomialFeatureAdder(degree=2)))
    return Pipeline(steps=steps)

def plot_correlation_matrix(df):
    df_numeric = df.select_dtypes(include=['number'])
    corr_matrix = df_numeric.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
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
