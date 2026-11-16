from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC

from preprocessing import load_and_preprocess_data
from vectorization import vectorize_text


def train_logistic_regression(X_train_tfidf, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    return model


def train_naive_bayes(X_train_tfidf, y_train):
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_tfidf, y_train)
    return model


def train_linear_svc(X_train_tfidf, y_train):
    model = LinearSVC(max_iter=5000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    return model


def train_complement_nb(X_train_tfidf, y_train):
    model = ComplementNB(alpha=1.0)
    model.fit(X_train_tfidf, y_train)
    return model


def main():
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "dataset" / "customer_support_tickets.csv"
    models_path = project_root / "models"

    models_path.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)

    vectorizer, X_train_tfidf, X_test_tfidf = vectorize_text(X_train, X_test)

    models = {
        "logistic_regression": train_logistic_regression(X_train_tfidf, y_train),
        "naive_bayes": train_naive_bayes(X_train_tfidf, y_train),
        "linear_svc": train_linear_svc(X_train_tfidf, y_train),
        "complement_nb": train_complement_nb(X_train_tfidf, y_train),
    }

    for model_name, model in models.items():
        joblib.dump(model, models_path / f"{model_name}.pkl")
        print(f"{model_name} training completed successfully!")
        print(f"Model saved: models/{model_name}.pkl")

    joblib.dump(vectorizer, models_path / "tfidf_vectorizer.pkl")

    print("\nAll models trained and saved successfully!")
    print(f"Training samples: {X_train_tfidf.shape[0]}")
    print(f"Testing samples: {X_test_tfidf.shape[0]}")
    print(f"Features: {X_train_tfidf.shape[1]}")
    print("Vectorizer saved: models/tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()
