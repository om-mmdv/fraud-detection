"""
Stratified train/test split for the fraud project.

Every model uses the same split (same random_state),
so scores can be compared fairly.
"""

from sklearn.model_selection import train_test_split

from src.data import TARGET, load_data

RANDOM_STATE = 42
TEST_SIZE = 0.2


def get_split():
    """
    Returns X_train, X_test, y_train, y_test.

    stratify=y keeps the same fraud share (0.173%) in both parts.
    The "Time" column is dropped: it is the number of seconds since the
    first transaction, which means nothing for new data.
    """
    df = load_data()
    X = df.drop(columns=[TARGET, "Time"])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_split()
    print(f"Train: {len(X_train)} rows, frauds: {y_train.sum()}")
    print(f"Test:  {len(X_test)} rows, frauds: {y_test.sum()}")