# feature_selection_pipeline.py (Updated)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

def compute_mutual_information(df, target_column='rainfall'):
    if 'windspeed_category' in df.columns:
        df['windspeed_category'] = df['windspeed_category'].fillna('Unknown')

    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column]

    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in X.columns:
        if X[col].isnull().sum() > 0:
            col_median = X[col].median()
            if np.isnan(col_median):
                X[col].fillna(0, inplace=True)
            else:
                X[col].fillna(col_median, inplace=True)

    X.fillna(0, inplace=True)

    mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
    mi_df = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores})
    mi_df = mi_df.sort_values(by='MI_Score', ascending=False)

    print("Top features by mutual information:")
    print(mi_df.head(10))
    return mi_df

def plot_mi_scores(mi_df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='MI_Score', y='Feature', data=mi_df, palette='viridis')
    plt.title('Mutual Information Scores')
    plt.xlabel('MI Score')
    plt.ylabel('Features')
    plt.show()

def drop_low_impact_features(df, mi_df, threshold=0.02, target_column='rainfall'):
    # Protect features containing these keywords
    protected_keywords = ['mintemp']

    # Select features to keep
    to_keep = mi_df[
        (mi_df['MI_Score'] >= threshold) |
        (mi_df['Feature'].str.contains('|'.join(protected_keywords)))
    ]['Feature'].tolist()

    if target_column not in to_keep:
        to_keep.append(target_column)

    df_refined = df[to_keep]
    print(f"Remaining features after filtering: {df_refined.columns.tolist()}")
    return df_refined

def save_refined_dataset(df, path):
    df.to_csv(path, index=False)
    print(f"Refined dataset saved at {path}")

if __name__ == "__main__":
    cleaned_train_path = "../Data/cleaned_train_with_features.csv"  
    df_cleaned = pd.read_csv(cleaned_train_path)

    if 'windspeed_category' in df_cleaned.columns:
        df_cleaned['windspeed_category'] = df_cleaned['windspeed_category'].fillna('Unknown')

    mi_df = compute_mutual_information(df_cleaned)
    plot_mi_scores(mi_df)

    df_refined = drop_low_impact_features(df_cleaned, mi_df, threshold=0.02)
    save_refined_dataset(df_refined, "../Data/refined_train.csv")
