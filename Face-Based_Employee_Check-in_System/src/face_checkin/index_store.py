import json
from pathlib import Path

import faiss


# ============================================================
# SAVE
# ============================================================

def save_index(index, metadata, index_path, metadata_path):
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# LOAD
# ============================================================

def load_index(index_path, metadata_path):
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    index = faiss.read_index(str(index_path))

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata