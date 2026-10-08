from spam_classifier.config import MODEL_NAME
from spam_classifier.data_loader import load_dataset, split_dataset
from spam_classifier.embedder import E5Embedder
from spam_classifier.vector_store import FaissVectorStore
from spam_classifier.classifier import KNNSpamClassifier
from spam_classifier.evaluator import evaluate_classifier, save_error_analysis


class SpamClassifierPipeline:

    def __init__(self):
        self.messages = None
        self.labels = None

        self.embedder = None
        self.vector_store = None
        self.classifier = None

        self.train_embeddings = None
        self.test_embeddings = None

        self.train_metadata = None
        self.test_metadata = None

    def build(self):
        """Build the complete spam classification pipeline."""

        print("Loading dataset...")

        _, self.messages, self.labels = load_dataset()

        print(f"Number of messages: {len(self.messages)}")
        print(f"Labels: {set(self.labels)}")

        self.embedder = E5Embedder(
            model_name=MODEL_NAME
        )

        print("Generating message embeddings...")

        embeddings = self.embedder.get_embeddings(
            self.messages,
            prefix="passage"
        )

        (
            self.train_embeddings,
            self.test_embeddings,
            self.train_metadata,
            self.test_metadata
        ) = split_dataset(
            self.messages,
            self.labels,
            embeddings
        )

        self.vector_store = FaissVectorStore()

        self.vector_store.build(
            self.train_embeddings
        )

        self.classifier = KNNSpamClassifier(
            embedder=self.embedder,
            vector_store=self.vector_store,
            train_metadata=self.train_metadata
        )

        print(
            f"FAISS vectors: "
            f"{self.vector_store.size}"
        )

        print("Pipeline ready.")

    def classify(self, text, k=3):
        """Classify a text message."""

        self._check_pipeline()

        return self.classifier.predict(
            text,
            k=k
        )

    def evaluate(self, k_values=(1, 3, 5)):
        """Evaluate the classifier."""

        self._check_pipeline()

        accuracy_results, error_results = evaluate_classifier(
            classifier=self.classifier,
            test_embeddings=self.test_embeddings,
            test_metadata=self.test_metadata,
            k_values=k_values
        )

        output_path = save_error_analysis(
            accuracy_results=accuracy_results,
            error_results=error_results,
            model_name=MODEL_NAME,
            test_size=len(self.test_embeddings)
        )

        return accuracy_results, output_path

    def _check_pipeline(self):
        if self.classifier is None:
            raise RuntimeError(
                "Pipeline has not been built. Run build() first."
            )