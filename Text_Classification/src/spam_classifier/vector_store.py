import faiss


class FaissVectorStore:

    def __init__(self):
        self.index = None

    def build(self, embeddings):
        """Create a FAISS index and add training embeddings."""

        embedding_dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(embedding_dim)

        self.index.add(embeddings.astype("float32"))

    def search(self, query_embedding, k=3):
        """Search for the top-k most similar vectors."""

        if self.index is None:
            raise RuntimeError("FAISS index has not been built.")

        k = min(k, self.index.ntotal)

        scores, indices = self.index.search(
            query_embedding.astype("float32"),
            k
        )

        return scores, indices

    @property
    def size(self):
        if self.index is None:
            return 0

        return self.index.ntotal