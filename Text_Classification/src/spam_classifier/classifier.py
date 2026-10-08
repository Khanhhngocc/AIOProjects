import numpy as np


def majority_vote(predictions):
    """Return the most common label."""

    unique_labels, counts = np.unique(
        predictions,
        return_counts=True
    )

    prediction = unique_labels[
        np.argmax(counts)
    ]

    label_distribution = {
        label: int(count)
        for label, count in zip(unique_labels, counts)
    }

    return prediction, label_distribution


class KNNSpamClassifier:

    def __init__(self, embedder, vector_store, train_metadata):
        self.embedder = embedder
        self.vector_store = vector_store
        self.train_metadata = train_metadata

    def predict(self, text, k=3):
        """Classify a new text message."""

        query_embedding = self.embedder.embed_batch(
            [text],
            prefix="query"
        )

        return self.predict_from_embedding(
            query_embedding,
            k=k
        )

    def predict_from_embedding(self, query_embedding, k=3):
        """Classify an already embedded message."""

        scores, indices = self.vector_store.search(
            query_embedding,
            k=k
        )

        predictions = []
        neighbors = []

        for i in range(len(indices[0])):
            neighbor_idx = int(indices[0][i])
            neighbor_score = float(scores[0][i])

            metadata = self.train_metadata[
                neighbor_idx
            ]

            predictions.append(
                metadata["label"]
            )

            neighbors.append({
                "index": metadata["index"],
                "message": metadata["message"],
                "label": metadata["label"],
                "score": neighbor_score
            })

        prediction, label_distribution = majority_vote(
            predictions
        )

        return {
            "prediction": prediction,
            "neighbors": neighbors,
            "label_distribution": label_distribution
        }