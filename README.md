
# Credit Card Fraud Detection


!\[Dashboard](assets/dashboard.png)


Supervised and unsupervised fraud detection on highly imbalanced data

(492 frauds in 284,807 transactions = 0.173%), with threshold selection

based on the cost of errors, and an interactive Streamlit dashboard.



Dataset: \[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

(ULB, Kaggle). The original CSV is not included. Features V1-V28 are

anonymized PCA components, so they cannot be interpreted.



\## Results (test set: 56,962 transactions, 98 frauds)



| Model | Precision | Recall | PR-AUC | ROC-AUC |

|---|---|---|---|---|

| Logistic regression (balanced) | 0.059 | 0.918 | 0.718 | 0.971 |

| Boosting, no weights | 0.292 | 0.643 | 0.374 | 0.808 |

| \*\*Boosting, soft weights\*\* | \*\*0.932\*\* | \*\*0.837\*\* | \*\*0.882\*\* | \*\*0.980\*\* |



Precision and recall at threshold 0.5. Accuracy is not reported: always

predicting "normal" gives 99.83%.



\## Threshold and risk analysis



The threshold was chosen with 5-fold stratified cross-validation on the

training set only (out-of-fold scores), then applied once to the test set.



| Threshold | Caught | False alarms | Recall | Precision | Net benefit |

|---|---|---|---|---|---|

| 0.5 (default) | 82 | 6 | 0.837 | 0.932 | $6,863 |

| \*\*0.04 (CV-chosen)\*\* | \*\*86\*\* | 19 | 0.878 | 0.819 | \*\*$8,720\*\* |



Lowering the threshold raised the net benefit by about 27% (assuming a false

alarm costs $10). Cross-validated PR-AUC is 0.855 ± 0.024 and test PR-AUC is

0.882, so the model is not overfitted to the test set.



!\[Net benefit](assets/net\_benefit.png)

## Supervised vs unsupervised



Five unsupervised detectors were compared (no labels used for fitting):



| Method | PR-AUC | ROC-AUC | Frauds in top 0.5% | Frauds in top 5% |

|---|---|---|---|---|

| Isolation Forest | 0.174 | 0.954 | 46.9% | 85.7% |

| Isolation Forest (log-amount, scaled) | 0.194 | 0.954 | 46.9% | 83.7% |

| Local Outlier Factor | 0.039 | 0.867 | 14.3% | 54.1% |

| PCA reconstruction error | 0.180 | 0.958 | 50.0% | 88.8% |

| Mahalanobis distance | 0.175 | 0.959 | 49.0% | 87.8% |

| \*\*Supervised boosting\*\* | \*\*0.882\*\* | \*\*0.980\*\* | \*\*90.8%\*\* | \*\*92.9%\*\* |



All unsupervised methods hit the same ceiling: they rank transactions by how

unusual they are, not by how fraudulent they are, and many legitimate

transactions are unusual. The differences among the four best are within

noise (98 frauds in the test set). Labels roughly double the frauds found when

analysts review only the top 0.5% of transactions.

\## Dashboard



```bash

streamlit run app.py

```



Choose the model, move the threshold and see caught frauds, false alarms and

money saved. It uses `data/test\_scores.csv` (model scores only, no features).



\## Installation



```bash

git clone https://github.com/om-mmdv/fraud-detection.git

cd fraud-detection

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

```



Put `creditcard.csv` into `data/`, then:



```bash

python -m src.explore

python -m src.baseline

python -m src.train

python -m src.risk

python -m src.anomaly

python -m src.export\_scores

```



\## Limitations



\- Only 98 frauds in the test set: metrics are noisy (one case = 1 pp of recall).

\- The false alarm cost ($10) is an assumption; "saved" assumes the full amount

&#x20; of a caught transaction is recovered.

\- Data is from 2013 and 1,081 duplicated rows were kept.

\- `Time` was dropped; hour-of-day patterns are approximate.

\- Early stopping was disabled for boosting; the unweighted model overfits.
- Differences of 2-3 points in PR-AUC are within the fold-to-fold spread

&#x20; (± 0.024), so they should not be read as real improvements.

