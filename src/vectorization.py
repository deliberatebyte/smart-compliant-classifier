from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_text(X_train, X_test, max_features=10000):

    vectorizer = TfidfVectorizer(
        max_features=max_features, ngram_range=(1, 2), min_df=2
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("TF-IDF vectorization completed successfully!")
    print(f"Training matrix shape: {X_train_tfidf.shape}")
    print(f"Testing matrix shape: {X_test_tfidf.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    print("\nSample vocabulary:")
    print(vectorizer.get_feature_names_out()[:15])

    return vectorizer, X_train_tfidf, X_test_tfidf


if __name__ == "__main__":

    from preprocessing import load_and_preprocess_data

    csv_path = Path("dataset/customer_support_tickets.csv")

    X_train, X_test, y_train, y_test = load_and_preprocess_data(csv_path)

    vectorizer, X_train_tfidf, X_test_tfidf = vectorize_text(X_train, X_test)
