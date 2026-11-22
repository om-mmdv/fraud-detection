"""
Exploratory data analysis for credit card fraud detection.

Run from the project folder:
    python -m src.explore

Prints key facts and saves plots to assets/.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from src.data import TARGET, load_data


def print_summary(df):
    """Class balance, a useless baseline and amount statistics."""
    fraud = df[TARGET].sum()
    print("=== Class balance ===")
    print(f"  Fraud: {fraud} of {len(df)} ({fraud / len(df) * 100:.3f}%)")
    print(f"  A model that always says 'normal' has accuracy "
          f"{(1 - fraud / len(df)) * 100:.2f}% and catches 0 frauds.")

    print("\n=== Missing values ===")
    print(f"  Total missing cells: {df.isna().sum().sum()}")
    print(f"  Duplicated rows: {df.duplicated().sum()}")

    print("\n=== Transaction amount by class ===")
    print(df.groupby(TARGET)["Amount"].describe().round(2))


def plot_class_balance(df):
    """Bar chart of class counts on a log scale (fraud is too small otherwise)."""
    counts = df[TARGET].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(["Normal", "Fraud"], counts.values, color=["#4c72b0", "#dd8452"])
    plt.yscale("log")
    plt.ylabel("Transactions (log scale)")
    plt.title("Class balance")
    for i, value in enumerate(counts.values):
        plt.text(i, value, f"{value:,}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig("assets/class_balance.png", dpi=150)
    plt.close()


def plot_amounts(df):
    """Histogram of amounts per class, limited to 500 for readability."""
    plt.figure(figsize=(8, 5))
    for label, name, color in [(0, "Normal", "#4c72b0"), (1, "Fraud", "#dd8452")]:
        values = df.loc[df[TARGET] == label, "Amount"]
        # density=True: compare shapes, since class sizes differ hugely
        plt.hist(values, bins=50, range=(0, 500), density=True,
                 alpha=0.6, label=name, color=color)
    plt.xlabel("Amount (capped at 500 on the plot)")
    plt.ylabel("Share of transactions")
    plt.title("Transaction amounts by class")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/amounts.png", dpi=150)
    plt.close()


def plot_fraud_by_hour(df):
    """Fraud rate by approximate hour of day (Time is seconds from the start)."""
    hour = (df["Time"] / 3600 % 24).astype(int)
    rate = df.groupby(hour)[TARGET].mean() * 100

    plt.figure(figsize=(9, 4))
    plt.bar(rate.index, rate.values, color="#dd8452")
    plt.xlabel("Approximate hour of day")
    plt.ylabel("Fraud rate, %")
    plt.title("Fraud rate by hour")
    plt.tight_layout()
    plt.savefig("assets/fraud_by_hour.png", dpi=150)
    plt.close()

    print("\n=== Fraud rate by hour: top 3 ===")
    print(rate.sort_values(ascending=False).head(3).round(2))


def plot_top_features(df):
    """
    Which V features differ most between classes?
    Measure: difference of class means divided by the overall std.
    """
    features = [c for c in df.columns if c.startswith("V")]
    means = df.groupby(TARGET)[features].mean()
    separation = ((means.loc[1] - means.loc[0]) / df[features].std()).abs()
    top = separation.sort_values(ascending=False).head(10)

    plt.figure(figsize=(7, 5))
    plt.barh(top.index[::-1], top.values[::-1], color="#4c72b0")
    plt.xlabel("Class separation (standardized mean difference)")
    plt.title("Features that separate fraud from normal")
    plt.tight_layout()
    plt.savefig("assets/top_features.png", dpi=150)
    plt.close()

    print("\n=== Top 5 separating features ===")
    print(top.head(5).round(2))


def main():
    os.makedirs("assets", exist_ok=True)
    df = load_data()

    print_summary(df)
    plot_class_balance(df)
    plot_amounts(df)
    plot_fraud_by_hour(df)
    plot_top_features(df)
    print("\nPlots saved to assets/")


if __name__ == "__main__":
    main()