import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings
import os  # For saving models safely

warnings.filterwarnings("ignore", category=UserWarning)

def save_model(model, path):
    # Ensure directory exists
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Save model
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")

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

def train_models(X_train, y_train):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=300, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0),
        'LightGBM': LGBMClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=300, random_state=42),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=300, random_state=42),
        'SVM': SVC(probability=True, kernel='rbf', random_state=42)
    }
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        models[name] = model
        save_model(model, f"./Data/{name.lower()}_model.pkl")  # ✅ Correct path
    return models

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
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

def build_stacking_ensemble(models):
    estimators = [(name, model) for name, model in models.items() if name != 'LogisticRegression']
    stack_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        passthrough=True,
        n_jobs=-1
    )
    return stack_model

if __name__ == "__main__":
    cleaned_train_path = "./Data/refined_train.csv"  
    df_cleaned = load_cleaned_data(cleaned_train_path)
    X_train, X_val, y_train, y_val = split_features_target(df_cleaned)

    X_train, y_train = apply_smote(X_train, y_train)

    models = train_models(X_train, y_train)

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        evaluate_model(model, X_val, y_val)

    print("\nTraining and evaluating stacking ensemble...")
    stack_model = build_stacking_ensemble(models)
    stack_model.fit(X_train, y_train)
    evaluate_model(stack_model, X_val, y_val)

    save_model(stack_model, "./Data/best_stacking_ensemble_model.pkl")
