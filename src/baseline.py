"""
Baseline models for fraud detection.

Run from the project folder:
    python -m src.baseline

Accuracy is useless here (99.83% for "always normal"), so we report
precision, recall and PR-AUC instead.
"""

from sklearn.dummy import DummyClassifier
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


def report(name, y_true, scores, y_pred):
    """Print metrics. `scores` are probabilities, `y_pred` are 0/1 labels."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)
    print(f"{name:22s} | precision {precision:5.3f} | recall {recall:5.3f} "
          f"| PR-AUC {pr_auc:5.3f} | ROC-AUC {roc_auc:5.3f}")


def main():
    X_train, X_test, y_train, y_test = get_split()

    # Baseline 1: always predict "normal"
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    report("Always normal", y_test, dummy.predict_proba(X_test)[:, 1],
           dummy.predict(X_test))

    # Baseline 2: logistic regression.
    # class_weight="balanced" makes mistakes on the rare class cost more.
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000,
                           random_state=RANDOM_STATE),
    )
    logistic.fit(X_train, y_train)
    report("Logistic regression", y_test, logistic.predict_proba(X_test)[:, 1],
           logistic.predict(X_test))

    print(f"\nRandom-guess PR-AUC would be about {y_test.mean():.4f}")


if __name__ == "__main__":
    main()