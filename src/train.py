"""
Train gradient boosting for fraud detection and compare with baselines.

Run from the project folder:
    python -m src.train

Saves the best model and its test scores to outputs/.
"""

import os

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.split import RANDOM_STATE, get_split


def soft_weights(y):
    """
    Mild sample weights: frauds get sqrt(imbalance ratio) times more weight.
    Full balancing (ratio ~578) tends to flood us with false alarms.
    """
    ratio = (y == 0).sum() / (y == 1).sum()
    return np.where(y == 1, np.sqrt(ratio), 1.0)


def report(name, y_true, scores):
    """Print metrics at the default 0.5 threshold, plus threshold-free ones."""
    y_pred = (scores >= 0.5).astype(int)
    print(f"{name:28s} | precision {precision_score(y_true, y_pred, zero_division=0):5.3f} "
          f"| recall {recall_score(y_true, y_pred):5.3f} "
          f"| PR-AUC {average_precision_score(y_true, scores):5.3f} "
          f"| ROC-AUC {roc_auc_score(y_true, scores):5.3f}")


def main():
    X_train, X_test, y_train, y_test = get_split()
    os.makedirs("outputs", exist_ok=True)

    models = {
        "Logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000,
                               random_state=RANDOM_STATE)),
        "Boosting (no weights)": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE),
        "Boosting (soft weights)": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE),
    }

    best_name, best_pr, best_model = None, -1, None
    for name, model in models.items():
        if name == "Boosting (soft weights)":
            model.fit(X_train, y_train, sample_weight=soft_weights(y_train))
        else:
            model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        report(name, y_test, scores)

        pr_auc = average_precision_score(y_test, scores)
        if pr_auc > best_pr:
            best_name, best_pr, best_model = name, pr_auc, model

    joblib.dump(best_model, "outputs/best_model.joblib")
    print(f"\nBest model by PR-AUC: {best_name} ({best_pr:.3f})")
    print("Saved to outputs/best_model.joblib")


if __name__ == "__main__":
    main()