from spam_classifier.pipeline import SpamClassifierPipeline


def display_prediction(result):
    """Display classification results."""

    print("\n" + "=" * 60)

    print(
        f"Prediction: "
        f"{result['prediction'].upper()}"
    )

    print(
        f"Label distribution: "
        f"{result['label_distribution']}"
    )

    print("\nTop neighbors:")

    for i, neighbor in enumerate(
        result["neighbors"],
        start=1
    ):
        message = neighbor["message"]

        if len(message) > 100:
            message = message[:100] + "..."

        print(
            f"{i}. "
            f"Label: {neighbor['label']} | "
            f"Score: {neighbor['score']:.4f}"
        )

        print(f"Message: {message}")

    print("=" * 60)


def classify_message(pipeline):
    """Handle user message classification."""

    message = input("\nEnter message: ").strip()

    if not message:
        print("Message cannot be empty.")
        return

    k_input = input("Enter k [default=3]: ").strip()

    try:
        k = int(k_input) if k_input else 3

        if k <= 0:
            raise ValueError

    except ValueError:
        print("k must be a positive integer.")
        return

    result = pipeline.classify(
        message,
        k=k
    )

    display_prediction(result)


def evaluate_model(pipeline):
    """Evaluate the model using different k values."""

    print(
        "\nEvaluating k = 1, 3, 5..."
    )

    results, output_path = pipeline.evaluate(
        k_values=(1, 3, 5)
    )

    print("\nAccuracy results:")

    for k, accuracy in results.items():
        print(
            f"k={k}: "
            f"{accuracy:.4f} "
            f"({accuracy * 100:.2f}%)"
        )

    print(
        f"\nError analysis saved to: "
        f"{output_path}"
    )


def main():
    """Run the command-line application."""

    print("=" * 60)
    print("SPAM CLASSIFIER")
    print("=" * 60)

    pipeline = SpamClassifierPipeline()

    try:
        pipeline.build()

    except Exception as error:
        print(
            f"\nFailed to build pipeline: "
            f"{error}"
        )
        return

    while True:
        print("\n1. Classify a message")
        print("2. Evaluate classifier")
        print("3. Exit")

        choice = input("\nChoose an option: ").strip()

        if choice == "1":
            classify_message(pipeline)

        elif choice == "2":
            evaluate_model(pipeline)

        elif choice == "3":
            print("Goodbye!")
            break

        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()