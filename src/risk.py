"""
Threshold selection and risk analysis.

Run from the project folder:
    python -m src.risk

For many thresholds, computes caught/missed frauds, false alarms and the
money saved, then picks the threshold with the best net benefit.

ASSUMPTION: a false alarm costs FALSE_ALARM_COST dollars (review work and
customer annoyance). This is a made-up number: change it to see the effect.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.split import get_split

FALSE_ALARM_COST = 5.0  # dollars per blocked honest transaction (assumption)


def threshold_table(y_true, scores, amounts, thresholds):
    """For each threshold, compute counts and the money outcome."""
    rows = []
    for t in thresholds:
        flagged = scores >= t
        tp = flagged & (y_true == 1)   # frauds caught
        fp = flagged & (y_true == 0)   # false alarms
        fn = ~flagged & (y_true == 1)  # frauds missed

        saved = amounts[tp].sum()
        missed = amounts[fn].sum()
        alarm_cost = fp.sum() * FALSE_ALARM_COST
        rows.append({
            "threshold": t,
            "caught": int(tp.sum()),
            "missed": int(fn.sum()),
            "false_alarms": int(fp.sum()),
            "recall": tp.sum() / max((y_true == 1).sum(), 1),
            "precision": tp.sum() / max(flagged.sum(), 1),
            "saved_$": saved,
            "missed_$": missed,
            "alarm_cost_$": alarm_cost,
            "net_benefit_$": saved - alarm_cost,
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs("assets", exist_ok=True)
    _, X_test, _, y_test = get_split()
    model = joblib.load("outputs/best_model.joblib")

    y = y_test.to_numpy()
    scores = model.predict_proba(X_test)[:, 1]
    amounts = X_test["Amount"].to_numpy()

    thresholds = np.round(np.arange(0.02, 0.99, 0.02), 2)
    table = threshold_table(y, scores, amounts, thresholds)
    table.to_csv("outputs/threshold_table.csv", index=False)

    best = table.loc[table["net_benefit_$"].idxmax()]
    default = table.loc[(table["threshold"] - 0.5).abs().idxmin()]

    print(f"Total fraud amount in the test set: ${amounts[y == 1].sum():,.0f}")
    print(f"False alarm cost (assumption): ${FALSE_ALARM_COST:.0f}\n")
    for title, row in [("Default threshold 0.5", default), ("Best threshold", best)]:
        print(f"=== {title}: {row['threshold']:.2f} ===")
        print(f"  Caught {row['caught']:.0f}, missed {row['missed']:.0f}, "
              f"false alarms {row['false_alarms']:.0f}")
        print(f"  Recall {row['recall']:.3f}, precision {row['precision']:.3f}")
        print(f"  Saved ${row['saved_$']:,.0f}, missed ${row['missed_$']:,.0f}, "
              f"alarm cost ${row['alarm_cost_$']:,.0f}")
        print(f"  Net benefit ${row['net_benefit_$']:,.0f}\n")

    # Plot 1: precision and recall versus threshold
    plt.figure(figsize=(8, 5))
    plt.plot(table["threshold"], table["recall"], label="Recall", color="#3b82f6")
    plt.plot(table["threshold"], table["precision"], label="Precision", color="#06b6d4")
    plt.axvline(best["threshold"], color="#f97316", linestyle="--", label="Best threshold")
    plt.xlabel("Threshold")
    plt.title("Precision and recall by threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/threshold_curves.png", dpi=150)
    plt.close()

    # Plot 2: net benefit versus threshold
    plt.figure(figsize=(8, 5))
    plt.plot(table["threshold"], table["net_benefit_$"], color="#3b82f6")
    plt.axvline(best["threshold"], color="#f97316", linestyle="--", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Net benefit, $")
    plt.title("Net benefit by threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/net_benefit.png", dpi=150)
    plt.close()
    print("Plots saved to assets/")


if __name__ == "__main__":
    main()