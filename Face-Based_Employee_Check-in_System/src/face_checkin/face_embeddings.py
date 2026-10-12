import cv2
import faiss
import numpy as np

from insightface.app import FaceAnalysis
from .config import FACE_VECTOR_DIM, INSIGHTFACE_MODEL_NAME, INSIGHTFACE_PROVIDERS


class FaceEmbeddingEngine:

    def __init__(self, model_name=INSIGHTFACE_MODEL_NAME, providers=INSIGHTFACE_PROVIDERS):
        self.model = FaceAnalysis(
            name=model_name,
            providers=providers,
            allowed_modules=["detection", "recognition"]
        )

        # ctx_id = -1 means CPU
        self.model.prepare(ctx_id=-1)


    # ========================================================
    # IMAGE -> FACE EMBEDDING
    # ========================================================

    def image_to_embedding(self, image_path):
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        faces = self.model.get(image)

        if not faces:
            raise ValueError(f"No face detected in {image_path}")

        # If multiple faces are detected, use the largest one
        face = max(faces, key=self._face_area)
        embedding = face.normed_embedding

        if embedding is None:
            raise ValueError(f"Cannot generate embedding for {image_path}")

        return np.ascontiguousarray(embedding, dtype=np.float32)


    # ========================================================
    # FACE AREA
    # ========================================================

    @staticmethod
    def _face_area(face):
        x1, y1, x2, y2 = face.bbox
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)

        return width * height


    # ========================================================
    # BUILD FAISS INDEX
    # ========================================================

    def build_index(self, records):
        index = faiss.IndexFlatIP(FACE_VECTOR_DIM)
        metadata = []

        for record in records:
            try:
                embedding = self.image_to_embedding(record["image_path"])
                index.add(embedding[np.newaxis, :])
                metadata.append(record)

            except Exception as e:
                print(f"Cannot process {record['image_path']}: {e}")

        return index, metadata


    # ========================================================
    # SEARCH FROM EXISTING EMBEDDING
    # ========================================================

    def search_embedding(self, query_embedding, index, metadata, k=5):
        if index.ntotal == 0:
            return []

        k = min(k, index.ntotal)

        similarities, indices = index.search(
            query_embedding[np.newaxis, :],
            k
        )

        results = []

        for similarity, idx in zip(similarities[0], indices[0]):
            record = metadata[int(idx)]

            results.append({
                "label": record["label"],
                "image_path": record["image_path"],
                "similarity": float(similarity),
                "index": int(idx)
            })

        return results


    # ========================================================
    # SEARCH FROM IMAGE
    # ========================================================

    def search(self, query_image_path, index, metadata, k=5):
        query_embedding = self.image_to_embedding(query_image_path)

        return self.search_embedding(
            query_embedding,
            index,
            metadata,
            k=k
        )


    # ========================================================
    # RECOGNITION
    # ========================================================

    def recognize(self, query_image_path, index, metadata, similarity_threshold):
        results = self.search(
            query_image_path,
            index,
            metadata,
            k=1
        )

        if not results:
            return None

        best_match = results[0]

        if best_match["similarity"] >= similarity_threshold:
            return best_match

        return None