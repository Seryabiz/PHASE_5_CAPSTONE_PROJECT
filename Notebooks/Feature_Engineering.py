import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class CapOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, columns, lower_quantile=0.01, upper_quantile=0.99):
        self.columns = columns
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.bounds = {}

    def fit(self, X, y=None):
        for col in self.columns:
            lower = X[col].quantile(self.lower_quantile)
            upper = X[col].quantile(self.upper_quantile)
            self.bounds[col] = (lower, upper)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.bounds.items():
            X[col] = X[col].clip(lower=lower, upper=upper)
        return X

class AddHumidityIndex(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['humidity_index'] = X['humidity'] * X['dewpoint']
        return X

class AddTemperatureDifference(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['temp_diff'] = X['maxtemp'] - X['mintemp']
        return X

class CategorizeWindspeed(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        bins = [-np.inf, 5, 15, np.inf]
        labels = ['Low', 'Medium', 'High']
        X['windspeed_category'] = pd.cut(X['windspeed'], bins=bins, labels=labels)
        return X

class CyclicalFeaturesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, column='day', period=7):
        self.column = column
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.loc[:, ~X.columns.duplicated()]
        if self.column in X.columns:
            X[self.column] = pd.to_numeric(X[self.column], errors='coerce')
            if X[self.column].isnull().all():
                print(f"⚠️ All values in '{self.column}' are NaN. Skipping cyclical encoding.")
                return X
            X[f'{self.column}_sin'] = np.sin(2 * np.pi * X[self.column] / self.period)
            X[f'{self.column}_cos'] = np.cos(2 * np.pi * X[self.column] / self.period)
            X.drop(columns=[self.column], inplace=True)
        return X

class PolynomialFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)

    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        self.numeric_cols = numeric_cols
        self.poly.fit(X[numeric_cols])
        self.feature_names = self.poly.get_feature_names_out(numeric_cols)
        return self

    def transform(self, X):
        X = X.copy()
        X[self.numeric_cols] = X[self.numeric_cols].fillna(X[self.numeric_cols].median())
        poly_features = self.poly.transform(X[self.numeric_cols])
        df_poly = pd.DataFrame(poly_features, columns=self.feature_names, index=X.index)
        X.drop(columns=self.numeric_cols, inplace=True)
        return pd.concat([X.reset_index(drop=True), df_poly.reset_index(drop=True)], axis=1)

def plot_correlation_matrix(df, top_k=50, annot=True, cmap='coolwarm', size=(14, 12)):
    df_numeric = df.select_dtypes(include=[np.number])
    corr_matrix = df_numeric.corr()
    if 'rainfall' in corr_matrix.columns:
        target_corr = corr_matrix['rainfall'].abs().sort_values(ascending=False)
        top_features = target_corr.index[:top_k]
        corr_matrix = corr_matrix.loc[top_features, top_features]
    plt.figure(figsize=size)
    sns.heatmap(corr_matrix, annot=annot, fmt=".2f", cmap=cmap, square=True)
    plt.title("Top Correlation Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def drop_highly_correlated_features(df, threshold=0.85, save_path=None):
    corr_matrix = df.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"🩹 Dropping highly correlated features: {to_drop}")
    df_dropped = df.drop(columns=to_drop, errors='ignore')
    if save_path:
        df_dropped.to_csv(save_path, index=False)
        print(f"✅ Cleaned dataset saved at: {save_path}")
    return df_dropped


def build_feature_engineering_pipeline(columns_to_cap):
    pipeline = Pipeline(steps=[
        ('cap_outliers', CapOutliers(columns=columns_to_cap)),
        ('add_temp_diff', AddTemperatureDifference()),
        ('add_humidity_index', AddHumidityIndex()),
        ('categorize_windspeed', CategorizeWindspeed()),
        ('add_polynomials', PolynomialFeatureAdder(degree=2))
    ])
    return pipeline
