# preprocessing.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

def load_dataset(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test

def rename_columns(df_train, df_test):
    df_train.rename(columns={'temparature': 'temperature'}, inplace=True)
    df_test.rename(columns={'temparature': 'temperature'}, inplace=True)

def handle_missing_values(df_test):
    df_test['winddirection'] = df_test['winddirection'].fillna(df_test['winddirection'].median())

def create_cyclical_features(df_train, df_test):
    df_train['day_sin'] = np.sin(2 * np.pi * df_train['day'] / 365)
    df_train['day_cos'] = np.cos(2 * np.pi * df_train['day'] / 365)
    df_test['day_sin'] = np.sin(2 * np.pi * df_test['day'] / 365)
    df_test['day_cos'] = np.cos(2 * np.pi * df_test['day'] / 365)
    df_train.drop(columns=['day'], inplace=True)
    df_test.drop(columns=['day'], inplace=True)

def create_temp_range(df_train, df_test):
    df_train['temp_range'] = df_train['maxtemp'] - df_train['mintemp']
    df_test['temp_range'] = df_test['maxtemp'] - df_test['mintemp']

def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def plot_outliers(df, column):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()

def scale_continuous_features(df_train, df_test, continuous_features):
    scaler = MinMaxScaler()
    df_train[continuous_features] = scaler.fit_transform(df_train[continuous_features])
    df_test[continuous_features] = scaler.transform(df_test[continuous_features])
    return scaler

def encode_categorical(df_train, df_test, column_name):
    df_train = pd.get_dummies(df_train, columns=[column_name], prefix=column_name)
    df_test = pd.get_dummies(df_test, columns=[column_name], prefix=column_name)
    missing_cols = set(df_train.columns) - set(df_test.columns)
    for col in missing_cols:
        df_test[col] = 0
    return df_train, df_test

def run_full_preprocessing(train_path, test_path):
    df_train, df_test = load_dataset(train_path, test_path)
    rainfall = df_train['rainfall'].copy()  # Store target variable
    df_train.drop(columns=['rainfall'], inplace=True)  # Drop before processing

    rename_columns(df_train, df_test)
    handle_missing_values(df_test)
    create_cyclical_features(df_train, df_test)
    create_temp_range(df_train, df_test)

    for col in ['windspeed', 'temperature']:
        print(f"Outliers in {col}:")
        print(detect_outliers(df_train, col))
        plot_outliers(df_train, col)

    scaler = scale_continuous_features(df_train, df_test, ['windspeed', 'temperature', 'maxtemp', 'mintemp', 'humidity'])
    df_train, df_test = encode_categorical(df_train, df_test, 'winddirection')

    # Reattach rainfall column
    df_train['rainfall'] = rainfall.values

    print("Preprocessing complete! Ready for model training 🚀")
    return df_train, df_test, scaler
