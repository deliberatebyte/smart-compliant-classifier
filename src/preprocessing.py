import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(csv_path, test_size=0.2, random_state=42):

    df = pd.read_csv(csv_path)

    required_columns = ["Ticket Subject", "Ticket Description", "Ticket Type"]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    initial_rows = len(df)

    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)

    df = df.dropna(subset=["Ticket Type"])

    df["Ticket Subject"] = df["Ticket Subject"].fillna("").astype(str).str.strip()
    df["Ticket Description"] = (
        df["Ticket Description"].fillna("").astype(str).str.strip()
    )
    df["Ticket Type"] = df["Ticket Type"].astype(str).str.strip()

    df = df[df["Ticket Type"] != ""]

    df["Complaint Text"] = (
        df["Ticket Subject"] + " " + df["Ticket Description"]
    ).str.strip()

    df = df[df["Complaint Text"] != ""]

    X = df["Complaint Text"]
    y = df["Ticket Type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Preprocessing completed successfully!")
    print(f"Original records: {initial_rows}")
    print(f"Records after cleaning: {len(df)}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    csv_path = Path("dataset/customer_support_tickets.csv")

    load_and_preprocess_data(csv_path)
