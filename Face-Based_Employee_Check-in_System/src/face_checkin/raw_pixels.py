from pathlib import Path

import faiss
import numpy as np
from PIL import Image, ImageOps

from .config import RAW_IMAGE_SIZE, RAW_VECTOR_DIM


# ============================================================
# IMAGE -> RAW PIXEL VECTOR
# ============================================================

def image_to_raw_vector(image_path):
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img = img.resize((RAW_IMAGE_SIZE, RAW_IMAGE_SIZE))

    img_array = np.asarray(img, dtype=np.float32)

    # [0, 255] -> [0, 1]
    img_array = img_array / 255.0

    # (300, 300, 3) -> (270000,)
    vector = img_array.reshape(-1)

    return np.ascontiguousarray(vector, dtype=np.float32)


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_raw_pixel_index(records):
    index = faiss.IndexFlatL2(RAW_VECTOR_DIM)
    metadata = []

    for record in records:
        try:
            vector = image_to_raw_vector(record["image_path"])

            # FAISS requires shape: (number_of_vectors, dimension)
            index.add(vector[np.newaxis, :])

            metadata.append(record)

        except Exception as e:
            print(f"Cannot process {record['image_path']}: {e}")

    return index, metadata


# ============================================================
# SEARCH
# ============================================================

def search_raw_pixels(query_image_path, index, metadata, k=5):
    if index.ntotal == 0:
        return []

    query_vector = image_to_raw_vector(query_image_path)
    k = min(k, index.ntotal)

    distances, indices = index.search(query_vector[np.newaxis, :], k)

    results = []

    for distance, idx in zip(distances[0], indices[0]):
        record = metadata[int(idx)]

        results.append({
            "label": record["label"],
            "image_path": record["image_path"],
            "distance": float(distance)
        })

    return results


# ============================================================
# RECOGNITION
# ============================================================

def recognize_employee_raw(
    query_image_path,
    index,
    metadata,
    distance_threshold
):
    results = search_raw_pixels(
        query_image_path,
        index,
        metadata,
        k=1
    )

    if not results:
        return None

    best_match = results[0]

    if best_match["distance"] <= distance_threshold:
        return best_match

    return None