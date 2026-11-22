"""
Export test-set scores for the dashboard.

Run from the project folder:
    python -m src.export_scores

Writes data/test_scores.csv: label, amount, supervised score, anomaly score.
The file is small enough to publish, so the dashboard works without the
150 MB original dataset.
"""

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.split import RANDOM_STATE, get_split


def main():
    X_train, X_test, _, y_test = get_split()

    model = joblib.load("outputs/best_model.joblib")
    forest = IsolationForest(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    forest.fit(X_train)

    out = pd.DataFrame({
        "label": y_test.to_numpy(),
        "amount": X_test["Amount"].to_numpy(),
        "supervised_score": model.predict_proba(X_test)[:, 1].round(5),
        "anomaly_score": (-forest.score_samples(X_test)).round(5),
    })
    out.to_csv("data/test_scores.csv", index=False)
    print(f"Saved {len(out)} rows to data/test_scores.csv")


if __name__ == "__main__":
    main()