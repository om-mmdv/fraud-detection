"""
Honest threshold selection with cross-validation.

Run from the project folder:
    python -m src.validate

1. 5-fold stratified CV on the TRAIN set gives out-of-fold scores
   (each row is scored by a model that never saw it).
2. The threshold is chosen on those out-of-fold scores only.
3. The chosen threshold is applied ONCE to the test set.

This removes the leak of the earlier analysis, where the threshold was
tuned on the test set itself.
"""

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from src.risk import FALSE_ALARM_COST
from src.split import RANDOM_STATE, get_split
from src.train import soft_weights

N_FOLDS = 5


def make_model():
    """Same configuration as the best model in train.py."""
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, early_stopping=False,
        random_state=RANDOM_STATE)


def net_benefit(y, scores, amounts, threshold):
    """Money saved on caught frauds minus the cost of false alarms."""
    flag = scores >= threshold
    saved = amounts[flag & (y == 1)].sum()
    alarms = (flag & (y == 0)).sum()
    return saved - alarms * FALSE_ALARM_COST


def main():
    X_train, X_test, y_train, y_test = get_split()
    amounts_train = X_train["Amount"].to_numpy()
    amounts_test = X_test["Amount"].to_numpy()
    y_tr, y_te = y_train.to_numpy(), y_test.to_numpy()

    # ---- Step 1: out-of-fold scores on the train set ----
    oof = np.zeros(len(X_train))
    fold_pr = []
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (fit_idx, val_idx) in enumerate(folds.split(X_train, y_tr), start=1):
        model = clone(make_model())
        model.fit(X_train.iloc[fit_idx], y_tr[fit_idx],
                  sample_weight=soft_weights(y_tr[fit_idx]))
        oof[val_idx] = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        pr = average_precision_score(y_tr[val_idx], oof[val_idx])
        fold_pr.append(pr)
        print(f"  Fold {fold}: PR-AUC {pr:.3f}")

    print(f"\nCV PR-AUC: {np.mean(fold_pr):.3f} +/- {np.std(fold_pr):.3f} "
          f"(spread across folds shows how noisy the estimate is)")

    # ---- Step 2: choose the threshold on out-of-fold scores ----
    thresholds = np.round(np.arange(0.02, 0.99, 0.02), 2)
    benefits = [net_benefit(y_tr, oof, amounts_train, t) for t in thresholds]
    best_t = thresholds[int(np.argmax(benefits))]
    print(f"\nThreshold chosen on validation (out-of-fold) data: {best_t}")

    # ---- Step 3: final model on the full train set, ONE test evaluation ----
    final = make_model()
    final.fit(X_train, y_tr, sample_weight=soft_weights(y_tr))
    test_scores = final.predict_proba(X_test)[:, 1]

    print("\n=== Test set (threshold fixed in advance) ===")
    for name, t in [("Default 0.5", 0.5), (f"CV-chosen {best_t}", best_t)]:
        flag = test_scores >= t
        print(f"  {name:18s} | recall {recall_score(y_te, flag):.3f} "
              f"| precision {precision_score(y_te, flag, zero_division=0):.3f} "
              f"| caught {int((flag & (y_te == 1)).sum())} "
              f"| false alarms {int((flag & (y_te == 0)).sum())} "
              f"| net benefit ${net_benefit(y_te, test_scores, amounts_test, t):,.0f}")
    print(f"  Test PR-AUC: {average_precision_score(y_te, test_scores):.3f}")


if __name__ == "__main__":
    main()