import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def cap_outliers(df, column, lower_quantile=0.05, upper_quantile=0.95):
    lower_bound = df[column].quantile(lower_quantile)
    upper_bound = df[column].quantile(upper_quantile)
    df[column] = np.clip(df[column], lower_bound, upper_bound)
    return df

def create_new_features(df):
    df['temp_diff'] = df['maxtemp'] - df['mintemp']
    df['humidity_index'] = df['humidity'] / df['temperature']
    df['windspeed_category'] = pd.cut(df['windspeed'], bins=[0, 20, 40, np.inf], labels=['Low', 'Medium', 'High'])
    return df

def split_data(df, target_column='rainfall'):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def check_data_types_and_uniques(df):
    print(df.dtypes)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        print(f"{col}: {df[col].unique()[:5]}")  # Show first 5 unique values

def convert_to_numeric(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

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