"""
Data loading for the credit card fraud detection project.

Every other script gets its data from load_data(),
so the loading logic lives in one place.
"""

import os

import pandas as pd

DATA_PATH = os.path.join("data", "creditcard.csv")
TARGET = "Class"  # 1 = fraud, 0 = normal transaction


def load_data():
    """Read creditcard.csv and return it as a pandas DataFrame."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"File not found: {DATA_PATH}\n"
            "Download creditcard.csv from Kaggle "
            "(search: 'Credit Card Fraud Detection') and put it into data/."
        )
    return pd.read_csv(DATA_PATH)


if __name__ == "__main__":
    # Quick self-check: run `python -m src.data`
    df = load_data()
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    counts = df[TARGET].value_counts()
    print("\nClass counts:")
    print(counts)
    print(f"Fraud share: {counts[1] / len(df) * 100:.3f}%")