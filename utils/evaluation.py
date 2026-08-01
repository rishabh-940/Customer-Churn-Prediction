import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# Cretae image folder automatically
def create_output_directory():

    os.makedirs("images", exist_ok=True)

# Calculate all metrics
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {

        "Accuracy": accuracy_score(y_test, y_pred),

        "Precision": precision_score(y_test, y_pred),

        "Recall": recall_score(y_test, y_pred),

        "F1 Score": f1_score(y_test, y_pred),

        "ROC AUC": roc_auc_score(y_test, y_prob)

    }

    print("\nModel Evaluation")
    print("=" * 50)

    for metric, value in metrics.items():
        print(f"{metric:<12}: {value:.4f}")

    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))

    return y_pred, y_prob, metrics

# Confusion matrix plot
def plot_confusion_matrix(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig("images/confusion_matrix.png")

    plt.close()
    
# ROC curve
def plot_roc_curve(y_test, y_prob):

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(6,5))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {auc:.3f}"
    )

    plt.plot([0,1],[0,1],"--")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig("images/roc_curve.png")

    plt.close()

# Feature importance
def plot_feature_importance(model, feature_names):

    importance = model.feature_importances_

    indices = np.argsort(importance)

    plt.figure(figsize=(10,8))

    plt.barh(
        range(len(indices)),
        importance[indices]
    )

    plt.yticks(
        range(len(indices)),
        np.array(feature_names)[indices]
    )

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig("images/feature_importance.png")

    plt.close()
