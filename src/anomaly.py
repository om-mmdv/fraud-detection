"""
Unsupervised outlier detection with Isolation Forest.

Run from the project folder:
    python -m src.anomaly

The model never sees the fraud labels. Labels are used only afterwards,
to measure how well "strangeness" matches real fraud, and to compare
with the supervised model.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.split import RANDOM_STATE, get_split


def recall_in_top(y_true, scores, share):
    """Share of all frauds found among the `share` most suspicious transactions."""
    n_top = int(len(scores) * share)
    top = np.argsort(scores)[::-1][:n_top]
    return y_true[top].sum() / y_true.sum()


def main():
    os.makedirs("assets", exist_ok=True)
    X_train, X_test, _, y_test = get_split()
    y = y_test.to_numpy()

    # Unsupervised: fit on train features only, no labels passed
    forest = IsolationForest(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    forest.fit(X_train)

    # score_samples: lower = more abnormal, so flip the sign
    anomaly_scores = -forest.score_samples(X_test)

    print("=== Isolation Forest (no labels used for training) ===")
    print(f"  PR-AUC:  {average_precision_score(y, anomaly_scores):.3f}")
    print(f"  ROC-AUC: {roc_auc_score(y, anomaly_scores):.3f}")

    # Review-budget view: if analysts check only the top X% most suspicious
    supervised = joblib.load("outputs/best_model.joblib")
    supervised_scores = supervised.predict_proba(X_test)[:, 1]

    print("\n=== Frauds found when checking only the top X% of transactions ===")
    print(f"  {'Top %':>6s} | {'Isolation Forest':>16s} | {'Supervised':>10s}")
    for share in [0.001, 0.005, 0.01, 0.05]:
        iso = recall_in_top(y, anomaly_scores, share)
        sup = recall_in_top(y, supervised_scores, share)
        print(f"  {share * 100:5.1f}% | {iso:16.3f} | {sup:10.3f}")

    print(f"\n  Supervised PR-AUC for comparison: "
          f"{average_precision_score(y, supervised_scores):.3f}")

    # Plot: distribution of anomaly scores per class
    plt.figure(figsize=(8, 5))
    plt.hist(anomaly_scores[y == 0], bins=60, density=True, alpha=0.6,
             label="Normal", color="#3b82f6")
    plt.hist(anomaly_scores[y == 1], bins=30, density=True, alpha=0.6,
             label="Fraud", color="#f97316")
    plt.xlabel("Anomaly score (higher = more unusual)")
    plt.ylabel("Density")
    plt.title("Isolation Forest scores by class")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/anomaly_scores.png", dpi=150)
    plt.close()
    print("\nPlot saved to assets/anomaly_scores.png")


if __name__ == "__main__":
    main()