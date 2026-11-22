"""
Fraud detection dashboard (Streamlit).

Run from the project folder:
    streamlit run app.py

Uses data/test_scores.csv (model scores on the test set).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BLUE, CYAN, ORANGE = "#3b82f6", "#06b6d4", "#f97316"

st.set_page_config(page_title="Fraud detection", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
    .card {background: linear-gradient(145deg,#131c2e,#0f1829);
           border:1px solid #22304a; border-radius:16px; padding:16px 18px;}
    .t {color:#8fa3c4; font-size:0.85rem;}
    .v {color:#e6edf7; font-size:1.8rem; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load():
    """Read the exported test-set scores once; Streamlit caches the result."""
    return pd.read_csv("data/test_scores.csv")


df = load()
y = df["label"].to_numpy()
amount = df["amount"].to_numpy()

# ---------- Sidebar ----------
st.sidebar.header("Settings")
model_name = st.sidebar.radio(
    "Model", ["Supervised (boosting)", "Isolation Forest (no labels)"])
is_supervised = model_name.startswith("Supervised")
scores = (df["supervised_score"] if is_supervised
          else df["anomaly_score"]).to_numpy()

# For the unsupervised model, the threshold is a percentile of suspiciousness
if is_supervised:
    threshold = st.sidebar.slider("Probability threshold", 0.01, 0.99, 0.08, 0.01)
else:
    top_share = st.sidebar.slider("Review the top % of transactions", 0.1, 10.0, 1.0, 0.1)
    threshold = np.quantile(scores, 1 - top_share / 100)

alarm_cost = st.sidebar.number_input(
    "False alarm cost, $ (assumption)", 0.0, 500.0, 10.0, 1.0)

# ---------- Metrics ----------
flag = scores >= threshold
tp = (flag & (y == 1)).sum()
fp = (flag & (y == 0)).sum()
saved = amount[flag & (y == 1)].sum()
missed = amount[~flag & (y == 1)].sum()
net = saved - fp * alarm_cost
recall = tp / max(y.sum(), 1)
precision = tp / max(flag.sum(), 1)

st.title("Credit Card Fraud Detection")
st.caption("Test set: 56,962 transactions, 98 fraudulent. "
           "Dataset: Credit Card Fraud (ULB, 2013).")


def card(col, title, value):
    """Render one KPI card in the given column."""
    col.markdown(f'<div class="card"><div class="t">{title}</div>'
                 f'<div class="v">{value}</div></div>', unsafe_allow_html=True)


c = st.columns(4)
card(c[0], "Frauds caught", f"{tp} of {int(y.sum())}")
card(c[1], "Recall / Precision", f"{recall:.0%} / {precision:.0%}")
card(c[2], "False alarms", f"{fp}")
card(c[3], "Net benefit", f"${net:,.0f}")

c = st.columns(2)
card(c[0], "Money saved", f"${saved:,.0f}")
card(c[1], "Money lost (missed frauds)", f"${missed:,.0f}")

# ---------- Net benefit curve ----------
st.subheader("Net benefit by threshold")
grid = np.quantile(scores, np.linspace(0.90, 0.9999, 120))
benefit = [amount[(scores >= t) & (y == 1)].sum()
           - ((scores >= t) & (y == 0)).sum() * alarm_cost for t in grid]
fig = go.Figure(go.Scatter(x=grid, y=benefit, line=dict(color=BLUE, width=2)))
fig.add_vline(x=threshold, line_color=ORANGE, line_dash="dash")
fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf7"),
                  xaxis_title="Threshold", yaxis_title="Net benefit, $",
                  margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig, use_container_width=True)

st.info("The threshold was tuned on the same data the metrics are computed on, "
        "and the test set contains only 98 frauds, so the numbers are noisy "
        "and optimistic. The false alarm cost is an assumption.")