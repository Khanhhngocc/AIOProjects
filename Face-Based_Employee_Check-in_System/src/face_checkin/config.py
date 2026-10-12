from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "raw" / "Dataset"
QUERY_DIR = DATA_DIR / "query"
INDEX_DIR = DATA_DIR / "processed" / "indexes"


# ============================================================
# INDEX PATHS
# ============================================================

RAW_INDEX_PATH = INDEX_DIR / "raw_pixels.index"
RAW_METADATA_PATH = INDEX_DIR / "raw_metadata.json"

FACE_INDEX_PATH = INDEX_DIR / "face_embeddings.index"
FACE_METADATA_PATH = INDEX_DIR / "face_metadata.json"


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ============================================================
# RAW PIXEL CONFIGURATION
# ============================================================

RAW_IMAGE_SIZE = 300
RAW_VECTOR_DIM = RAW_IMAGE_SIZE * RAW_IMAGE_SIZE * 3


# ============================================================
# INSIGHTFACE CONFIGURATION
# ============================================================

FACE_VECTOR_DIM = 512

INSIGHTFACE_MODEL_NAME = "buffalo_l"
INSIGHTFACE_PROVIDERS = ["CPUExecutionProvider"]