import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load cleaned and engineered dataset
df = pd.read_csv('./Data/cleaned_train_with_features.csv')

# Separate features and target
X = df.drop(columns=['rainfall'])
y = df['rainfall']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Define aggressive hyperparameter grids
random_forest_params = {
    'n_estimators': [200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

xgb_params = {
    'n_estimators': [300, 500],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1]
}

gb_params = {
    'n_estimators': [300, 500],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

lgb_params = {
    'n_estimators': [300, 500],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7]
}

svc_params = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['rbf']
}

# Hyperparameter tuning function
def tune_model(model, param_grid, X_train, y_train, model_name):
    print(f"\nTuning {model_name}...")
    grid = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, verbose=2, random_state=42)
    grid.fit(X_train, y_train)
    print(f"Best params for {model_name}: {grid.best_params_}")
    return grid.best_estimator_

# Tune top models
best_rf = tune_model(RandomForestClassifier(random_state=42), random_forest_params, X_train, y_train, 'RandomForest')
best_xgb = tune_model(XGBClassifier(eval_metric='logloss', random_state=42), xgb_params, X_train, y_train, 'XGBoost')
best_gb = tune_model(GradientBoostingClassifier(random_state=42), gb_params, X_train, y_train, 'GradientBoosting')
best_lgb = tune_model(LGBMClassifier(random_state=42), lgb_params, X_train, y_train, 'LightGBM')
best_svc = tune_model(SVC(probability=True, random_state=42), svc_params, X_train, y_train, 'SVC')

# Build Stacking Ensemble
stacking_ensemble = StackingClassifier(
    estimators=[
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('gb', best_gb),
        ('lgb', best_lgb),
        ('svc', best_svc)
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1,
    passthrough=True
)

# Train ensemble
stacking_ensemble.fit(X_train, y_train)

# Evaluate ensemble on validation set
y_pred = stacking_ensemble.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"\nStacking Ensemble Accuracy: {acc:.4f}")
print("Classification Report:\n", classification_report(y_val, y_pred))

# Save the final ensemble model
joblib.dump(stacking_ensemble, './Data/best_stacking_ensemble_tuned.pkl')
print("\n✅ Tuned stacking ensemble saved!")
