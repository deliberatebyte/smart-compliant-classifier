from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression

from preprocessing import load_and_preprocess_data
from vectorization import vectorize_text


def train_logistic_regression(X_train_tfidf, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)

    model.fit(X_train_tfidf, y_train)

    return model


def main():
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "dataset" / "customer_support_tickets.csv"
    models_path = project_root / "models"

    models_path.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)

    vectorizer, X_train_tfidf, X_test_tfidf = vectorize_text(X_train, X_test)

    logistic_model = train_logistic_regression(X_train_tfidf, y_train)

    joblib.dump(logistic_model, models_path / "logistic_regression.pkl")

    joblib.dump(vectorizer, models_path / "tfidf_vectorizer.pkl")

    print("Logistic Regression training completed successfully!")
    print(f"Training samples: {X_train_tfidf.shape[0]}")
    print(f"Features: {X_train_tfidf.shape[1]}")
    print("Model saved: models/logistic_regression.pkl")
    print("Vectorizer saved: models/tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()
