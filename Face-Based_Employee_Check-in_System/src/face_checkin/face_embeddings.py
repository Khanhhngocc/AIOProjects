from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN, InceptionResnetV1

from .config import DEVICE, FACE_IMAGE_SIZE, FACE_VECTOR_DIM


class FaceEmbeddingEngine:

    def __init__(self, device=DEVICE):
        self.device = device

        # Face detector
        self.face_detector = MTCNN(
            image_size=FACE_IMAGE_SIZE,
            margin=0,
            keep_all=False,
            post_process=True,
            device=self.device
        )

        # Face embedding model
        self.face_embedding_model = (
            InceptionResnetV1(pretrained="vggface2")
            .eval()
            .to(self.device)
        )


    # ========================================================
    # IMAGE -> FACE EMBEDDING
    # ========================================================

    def image_to_embedding(self, image_path):
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        # Detect + crop + preprocess face
        face_tensor = self.face_detector(img)

        if face_tensor is None:
            raise ValueError(f"No face detected in {image_path}")

        # (3, 160, 160) -> (1, 3, 160, 160)
        face_tensor = face_tensor.unsqueeze(0).to(self.device)

        # Generate FaceNet embedding
        with torch.inference_mode():
            embedding = self.face_embedding_model(face_tensor)

        embedding = (
            embedding
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # Inner Product becomes equivalent to cosine similarity
        faiss.normalize_L2(embedding)

        return np.ascontiguousarray(embedding[0])


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
    # SEARCH
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

    def recognize(
        self,
        query_image_path,
        index,
        metadata,
        similarity_threshold
    ):
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

    # ============================================================
    # SEARCH FROM EXISTING EMBEDDING
    # ============================================================

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