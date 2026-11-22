
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



Lowering the threshold from 0.5 to 0.08 caught 3 more frauds (85 vs 82)

at the price of 8 more false alarms, raising the net benefit by about 27%

(assuming a false alarm costs $10; the conclusion held for $1 and $5).



!\[Net benefit](assets/net\_benefit.png)



\## Supervised vs unsupervised



Isolation Forest never sees labels (PR-AUC 0.174, ROC-AUC 0.954). If analysts

review only the top 0.5% of transactions, it finds 47% of frauds versus 91%

for the supervised model; at 5% it finds 86% versus 93%.



!\[Anomaly scores](assets/anomaly\_scores.png)



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

\- The threshold was tuned on the same test set, so the benefit is optimistic;

&#x20; a separate validation split would be more correct.

\- The false alarm cost ($10) is an assumption; "saved" assumes the full amount

&#x20; of a caught transaction is recovered.

\- Data is from 2013 and 1,081 duplicated rows were kept.

\- `Time` was dropped; hour-of-day patterns are approximate.

\- Early stopping was disabled for boosting; the unweighted model overfits.

