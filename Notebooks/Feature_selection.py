import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

def compute_mutual_information(df, target_column='rainfall'):
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column]
    
    # Replace Inf and NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.max(), inplace=True)

    mi_scores = mutual_info_classif(X, y, discrete_features='auto')
    mi_df = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores})
    mi_df = mi_df.sort_values(by='MI_Score', ascending=False)
    return mi_df

def plot_mi_scores(mi_df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='MI_Score', y='Feature', data=mi_df, palette='viridis')
    plt.title('Mutual Information Scores')
    plt.xlabel('MI Score')
    plt.ylabel('Features')
    plt.show()

def drop_low_impact_features(df, mi_df, threshold=0.01, target_column='rainfall'):
    low_impact_features = mi_df[mi_df['MI_Score'] < threshold]['Feature'].tolist()
    print(f"Dropping low impact features (MI < {threshold}):", low_impact_features)

    wind_dir_features = [col for col in df.columns if 'winddir' in col]
    print(f"Dropping wind direction features:", wind_dir_features)

    df_refined = df.drop(columns=low_impact_features + wind_dir_features, errors='ignore')
    
    return df_refined

def save_refined_dataset(df, path):
    df.to_csv(path, index=False)
    print(f"Refined dataset saved at {path}")

if __name__ == "__main__":
    cleaned_train_path = "../Data/cleaned_train.csv"  
    df_cleaned = pd.read_csv(cleaned_train_path)
    
    mi_df = compute_mutual_information(df_cleaned)
    plot_mi_scores(mi_df)
    
    df_refined = drop_low_impact_features(df_cleaned, mi_df, threshold=0.01)
    save_refined_dataset(df_refined, "../Data/refined_train.csv")
