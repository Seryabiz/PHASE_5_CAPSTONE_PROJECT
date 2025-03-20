import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def load_cleaned_data(path):
    return pd.read_csv(path)

def clean_feature_data(X):
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    X = X.drop(columns=non_numeric_cols, errors='ignore')
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.clip(lower=-1e6, upper=1e6)
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            col_median = X[col].median()
            if np.isnan(col_median):
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(col_median)
    return X

def split_features_target(df, target='rainfall'):
    X = df.drop(columns=[target], errors='ignore')
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = clean_feature_data(X_train)
    X_val = clean_feature_data(X_val)
    return X_train, X_val, y_train, y_val

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print("✅ Applied SMOTE. New class distribution:")
    print(y_resampled.value_counts())
    return X_resampled, y_resampled

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_bagging_classifier(X_train, y_train):
    model = BaggingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print("Classification Report:\n", classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    print(f"Accuracy Score: {acc:.4f}")
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)
        print("ROC AUC Score:", auc)
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

def tune_xgboost(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7]
    }
    grid = GridSearchCV(XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0),
                        param_grid, cv=3, scoring='roc_auc', verbose=0)
    grid.fit(X_train, y_train)
    print("Best Parameters:", grid.best_params_)
    return grid.best_estimator_

def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    cleaned_train_path = "../Data/cleaned_train.csv"  
    df_cleaned = load_cleaned_data(cleaned_train_path)
    X_train, X_val, y_train, y_val = split_features_target(df_cleaned)

    X_train, y_train = apply_smote(X_train, y_train)

    print("Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    evaluate_model(lr_model, X_val, y_val)

    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_val, y_val)

    print("Training Bagging Classifier...")
    bag_model = train_bagging_classifier(X_train, y_train)
    evaluate_model(bag_model, X_val, y_val)

    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_model(xgb_model, X_val, y_val)

    print("Tuning XGBoost hyperparameters...")
    tuned_xgb = tune_xgboost(X_train, y_train)
    evaluate_model(tuned_xgb, X_val, y_val)

    save_model(tuned_xgb, "../Data/best_xgboost_model.pkl")
