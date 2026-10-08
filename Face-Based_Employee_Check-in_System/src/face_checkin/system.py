from pathlib import Path

from .config import (
    DATASET_DIR,
    DEVICE,
    RAW_INDEX_PATH,
    RAW_METADATA_PATH,
    FACE_INDEX_PATH,
    FACE_METADATA_PATH
)

from .dataset import load_dataset

from .raw_pixels import (
    build_raw_pixel_index,
    search_raw_pixels,
    recognize_employee_raw
)

from .face_embeddings import FaceEmbeddingEngine
from .index_store import save_index, load_index


class FaceCheckinSystem:

    def __init__(self, dataset_dir=DATASET_DIR):
        self.dataset_dir = Path(dataset_dir)
        self.device = DEVICE

        self._face_engine = None

        self.raw_index = None
        self.raw_metadata = None

        self.face_index = None
        self.face_metadata = None


    # ========================================================
    # FACE ENGINE
    # ========================================================

    @property
    def face_engine(self):
        if self._face_engine is None:
            print(f"Loading MTCNN and FaceNet on {self.device}...")

            self._face_engine = FaceEmbeddingEngine(
                device=self.device
            )

        return self._face_engine


    # ========================================================
    # DATASET
    # ========================================================

    def get_records(self):
        return load_dataset(self.dataset_dir)


    # ========================================================
    # BUILD INDEXES
    # ========================================================

    def build_indexes(self, save=True):
        records = self.get_records()

        if not records:
            raise ValueError(
                f"No employee images found in {self.dataset_dir}"
            )

        print(f"Using device: {self.device}")
        print(f"Number of employee images: {len(records)}")

        # Raw pixel index
        print("\nBuilding raw-pixel index...")

        self.raw_index, self.raw_metadata = (
            build_raw_pixel_index(records)
        )

        # Face embedding index
        print("\nBuilding face-embedding index...")

        self.face_index, self.face_metadata = (
            self.face_engine.build_index(records)
        )

        if save:
            save_index(
                self.raw_index,
                self.raw_metadata,
                RAW_INDEX_PATH,
                RAW_METADATA_PATH
            )

            save_index(
                self.face_index,
                self.face_metadata,
                FACE_INDEX_PATH,
                FACE_METADATA_PATH
            )

        return {
            "raw_vectors": self.raw_index.ntotal,
            "face_embeddings": self.face_index.ntotal
        }


    # ========================================================
    # LOAD INDEXES
    # ========================================================

    def load_indexes(self):
        self.raw_index, self.raw_metadata = load_index(
            RAW_INDEX_PATH,
            RAW_METADATA_PATH
        )

        self.face_index, self.face_metadata = load_index(
            FACE_INDEX_PATH,
            FACE_METADATA_PATH
        )


    # ========================================================
    # ENSURE INDEXES ARE LOADED
    # ========================================================

    def _ensure_raw_index(self):
        if self.raw_index is None or self.raw_metadata is None:
            self.raw_index, self.raw_metadata = load_index(
                RAW_INDEX_PATH,
                RAW_METADATA_PATH
            )


    def _ensure_face_index(self):
        if self.face_index is None or self.face_metadata is None:
            self.face_index, self.face_metadata = load_index(
                FACE_INDEX_PATH,
                FACE_METADATA_PATH
            )


    # ========================================================
    # RAW SEARCH
    # ========================================================

    def search_raw(self, query_image_path, k=5):
        self._ensure_raw_index()

        return search_raw_pixels(
            query_image_path,
            self.raw_index,
            self.raw_metadata,
            k=k
        )


    # ========================================================
    # FACE SEARCH
    # ========================================================

    def search_face(self, query_image_path, k=5):
        self._ensure_face_index()

        return self.face_engine.search(
            query_image_path,
            self.face_index,
            self.face_metadata,
            k=k
        )


    # ========================================================
    # RAW RECOGNITION
    # ========================================================

    def recognize_raw(
        self,
        query_image_path,
        distance_threshold
    ):
        self._ensure_raw_index()

        return recognize_employee_raw(
            query_image_path,
            self.raw_index,
            self.raw_metadata,
            distance_threshold
        )


    # ========================================================
    # FACE RECOGNITION
    # ========================================================

    def recognize_face(
        self,
        query_image_path,
        similarity_threshold
    ):
        self._ensure_face_index()

        return self.face_engine.recognize(
            query_image_path,
            self.face_index,
            self.face_metadata,
            similarity_threshold
        )

    # ========================================================
    # FACE SEARCH FROM EXISTING EMBEDDING
    # ========================================================

    def search_face_embedding(self, query_embedding, k=5):
        self._ensure_face_index()

        return self.face_engine.search_embedding(
            query_embedding,
            self.face_index,
            self.face_metadata,
            k=k
        )