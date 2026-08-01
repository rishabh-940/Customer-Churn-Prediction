import numpy as np

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# Defining the models
def get_models():
    """
    Returns all models to compare.
    """

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        )

    }

    return models

# Cross Validation
def compare_models(models, X_train, y_train, cv=5):
    """
    Compare multiple models using cross-validation.
    """

    results = {}

    print("=" * 60)
    print("Cross Validation Results")
    print("=" * 60)

    for name, model in models.items():

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )

        results[name] = {

            "Mean Accuracy": scores.mean(),

            "Std": scores.std(),

            "Scores": scores

        }

        print(f"\n{name}")

        print(f"Fold Scores : {np.round(scores,4)}")

        print(f"Mean Accuracy : {scores.mean():.4f}")

        print(f"Std : {scores.std():.4f}")

    return results

# Find the best model
def select_best_model(results):

    best_model = max(
        results,
        key=lambda model: results[model]["Mean Accuracy"]
    )

    print("\n" + "="*60)

    print("Best Model :", best_model)

    print(
        "Accuracy :",
        round(results[best_model]["Mean Accuracy"],4)
    )

    return best_model

import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Training the selected model
def train_model(model_name, models, X_train, y_train):
    """
    Train the selected model.
    """

    model = models[model_name]

    model.fit(X_train, y_train)

    return model

# Evalute the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    """

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

    if y_prob is not None:
        print(f"ROC AUC  : {roc_auc_score(y_test, y_prob):.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None
    }

# Save the model
def save_model(model, filepath):
    """
    Save trained model.
    """

    joblib.dump(model, filepath)

    print(f"\nModel saved to {filepath}")

# Save the encoders
def save_encoders(encoders, filepath):
    """
    Save label encoders.
    """

    joblib.dump(encoders, filepath)

    print(f"Encoders saved to {filepath}")

# Load Model
def load_model(filepath):
    """
    Load saved model.
    """

    return joblib.load(filepath)

# Hyperparameter tuning

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

def tune_random_forest(X_train, y_train):

    param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5),
        "max_features": ["sqrt", "log2", None]
    }

    rf = RandomForestClassifier(random_state=42)

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1",
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(random_search.best_params_)

    print("\nBest CV F1 Score:")
    print(f"{random_search.best_score_:.4f}")

    return random_search.best_estimator_

