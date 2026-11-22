"""
Compare several unsupervised outlier detectors.

Run from the project folder:
    python -m src.anomaly_compare

No labels are used for fitting or feature selection. Labels are used only
at the end, to measure how well each "strangeness" score matches real fraud.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.split import RANDOM_STATE, get_split

SUBSAMPLE = 30_000  # LOF and PCA are slow on 228k rows


def prepare(X_train, X_test):
    """
    Log-transform Amount (it is extremely skewed) and standardize everything.
    The scaler is fitted on train only.
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    X_train["Amount"] = np.log1p(X_train["Amount"])
    X_test["Amount"] = np.log1p(X_test["Amount"])
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


def recall_in_top(y, scores, share):
    """Share of all frauds found among the `share` most suspicious rows."""
    n_top = int(len(scores) * share)
    top = np.argsort(scores)[::-1][:n_top]
    return y[top].sum() / y.sum()


def main():
    X_train, X_test, _, y_test = get_split()
    y = y_test.to_numpy()

    rng = np.random.default_rng(RANDOM_STATE)
    sample = rng.choice(len(X_train), size=SUBSAMPLE, replace=False)

    train_s, test_s = prepare(X_train, X_test)  # scaled, log Amount
    results = {}

    # 1. Isolation Forest on raw features (the earlier baseline)
    forest = IsolationForest(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    forest.fit(X_train)
    results["Isolation Forest (raw)"] = -forest.score_samples(X_test)

    # 2. Isolation Forest on log-Amount, standardized features
    forest2 = IsolationForest(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    forest2.fit(train_s)
    results["Isolation Forest (scaled)"] = -forest2.score_samples(test_s)

    # 3. Local Outlier Factor (novelty=True allows scoring new data)
    lof = LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=-1)
    lof.fit(train_s[sample])
    results["Local Outlier Factor"] = -lof.score_samples(test_s)

    # 4. PCA reconstruction error: how badly is a point rebuilt from
    #    a few components learned on (mostly) normal data
    pca = PCA(n_components=10, random_state=RANDOM_STATE)
    pca.fit(train_s[sample])
    rebuilt = pca.inverse_transform(pca.transform(test_s))
    results["PCA reconstruction error"] = ((test_s - rebuilt) ** 2).sum(axis=1)

    # 5. Mahalanobis distance from the center of the training data
    center = train_s[sample].mean(axis=0)
    inv_cov = np.linalg.pinv(np.cov(train_s[sample], rowvar=False))
    diff = test_s - center
    results["Mahalanobis distance"] = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)

    print(f"{'Method':28s} | PR-AUC | ROC-AUC | found in top 0.5% | top 5%")
    for name, scores in results.items():
        print(f"{name:28s} | {average_precision_score(y, scores):6.3f} "
              f"| {roc_auc_score(y, scores):7.3f} "
              f"| {recall_in_top(y, scores, 0.005):17.3f} "
              f"| {recall_in_top(y, scores, 0.05):6.3f}")


if __name__ == "__main__":
    main()