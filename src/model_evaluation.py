import joblib
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, cast
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)

from preprocessing import load_and_preprocess_data


def prepare_evaluation_data(project_root):

    dataset_path = project_root / "dataset" / "customer_support_tickets.csv"

    _, X_test, _, y_test = load_and_preprocess_data(dataset_path)

    vectorizer_path = project_root / "models" / "tfidf_vectorizer.pkl"
    vectorizer = joblib.load(vectorizer_path)

    X_test_tfidf = vectorizer.transform(X_test)

    return X_test_tfidf, y_test


def load_trained_models(project_root):

    models_path = project_root / "models"

    return {
        "Logistic Regression": joblib.load(models_path / "logistic_regression.pkl"),
        "Multinomial Naive Bayes": joblib.load(models_path / "naive_bayes.pkl"),
        "LinearSVC": joblib.load(models_path / "linear_svc.pkl"),
        "Complement Naive Bayes": joblib.load(models_path / "complement_nb.pkl"),
    }


def evaluate_model(model, X_test, y_test, model_name):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    report_dict = cast(
        dict[str, Any],
        classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    )

    print("\n" + "=" * 65)
    print(f"{model_name} Evaluation")
    print("=" * 65)

    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nClassification Report:")
    print(report)

    metrics = {
        "Accuracy": accuracy,
        "Macro Precision": report_dict["macro avg"]["precision"],
        "Macro Recall": report_dict["macro avg"]["recall"],
        "Macro F1": report_dict["macro avg"]["f1-score"],
        "Weighted Precision": report_dict["weighted avg"]["precision"],
        "Weighted Recall": report_dict["weighted avg"]["recall"],
        "Weighted F1": report_dict["weighted avg"]["f1-score"],
    }

    return predictions, metrics


def generate_confusion_matrices(y_test, predictions, results_path):

    labels = sorted(y_test.unique())

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    axes = axes.flatten()

    for ax, (model_name, model_predictions) in zip(axes, predictions.items()):

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            model_predictions,
            labels=labels,
            display_labels=labels,
            cmap="Blues",
            xticks_rotation=45,
            ax=ax,
            colorbar=False,
        )

        ax.set_title(f"{model_name} Confusion Matrix")
        ax.set_xlabel("Predicted Category")
        ax.set_ylabel("Actual Category")

    plt.tight_layout()

    plot_path = results_path / "confusion_matrices.png"

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nConfusion matrices saved to: {plot_path}")


def compare_models(evaluation_results, results_path):

    comparison = pd.DataFrame(evaluation_results).T

    print("\n" + "=" * 85)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 85)

    print(comparison.round(4).to_string())

    best_model = comparison["Macro F1"].idxmax()

    print("\n" + "=" * 65)
    print(f"Best Performing Model (Macro F1): {best_model}")
    print("=" * 65)

    comparison_path = results_path / "model_comparison.csv"

    comparison.to_csv(comparison_path)

    print(f"\nModel comparison saved to: {comparison_path}")


def main():

    project_root = Path(__file__).resolve().parent.parent

    results_path = project_root / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("SMART COMPLAINT CLASSIFIER - MODEL EVALUATION")
    print("=" * 65)

    X_test_tfidf, y_test = prepare_evaluation_data(project_root)

    print("\nEvaluation data prepared successfully!")
    print(f"Testing samples: {len(y_test)}")
    print(f"Testing feature matrix: {X_test_tfidf.shape}")

    models = load_trained_models(project_root)

    print("\nAll four saved models loaded successfully!")

    predictions = {}
    evaluation_results = {}

    for model_name, model in models.items():

        predictions[model_name], evaluation_results[model_name] = evaluate_model(
            model,
            X_test_tfidf,
            y_test,
            model_name,
        )

    generate_confusion_matrices(
        y_test,
        predictions,
        results_path,
    )

    compare_models(
        evaluation_results,
        results_path,
    )

    print("\nModel evaluation completed successfully!")


if __name__ == "__main__":
    main()
