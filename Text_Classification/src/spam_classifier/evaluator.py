import json

from datetime import datetime
from tqdm import tqdm

from spam_classifier.config import ERROR_ANALYSIS_PATH


def evaluate_classifier(
    classifier,
    test_embeddings,
    test_metadata,
    k_values=(1, 3, 5)
):
    """Evaluate KNN accuracy for different k values."""

    results = {}
    all_errors = {}

    for k in k_values:
        correct = 0
        errors = []

        for i in tqdm(
            range(len(test_embeddings)),
            desc=f"Evaluating k={k}"
        ):
            query_embedding = test_embeddings[
                i:i + 1
            ]

            result = classifier.predict_from_embedding(
                query_embedding,
                k=k
            )

            predicted_label = result["prediction"]
            true_label = test_metadata[i]["label"]

            if predicted_label == true_label:
                correct += 1

            else:
                errors.append({
                    "index": i,
                    "original_index": test_metadata[i]["index"],
                    "message": test_metadata[i]["message"],
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "neighbors": result["neighbors"],
                    "label_distribution": result["label_distribution"]
                })

        total = len(test_embeddings)
        accuracy = correct / total

        results[k] = accuracy
        all_errors[k] = errors

        print(
            f"k={k} | Accuracy: {accuracy:.4f} "
            f"({accuracy * 100:.2f}%) | "
            f"Errors: {len(errors)}/{total}"
        )

    return results, all_errors


def save_error_analysis(
    accuracy_results,
    error_results,
    model_name,
    test_size,
    output_path=ERROR_ANALYSIS_PATH
):
    """Save evaluation errors to a JSON file."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    error_analysis = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "test_size": test_size,
        "accuracy_results": {
            str(k): accuracy
            for k, accuracy in accuracy_results.items()
        },
        "errors_by_k": {}
    }

    for k, errors in error_results.items():
        error_analysis["errors_by_k"][f"k_{k}"] = {
            "total_errors": len(errors),
            "error_rate": len(errors) / test_size,
            "errors": errors
        }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            error_analysis,
            file,
            ensure_ascii=False,
            indent=2
        )

    return output_path