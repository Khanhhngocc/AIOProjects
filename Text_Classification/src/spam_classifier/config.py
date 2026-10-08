from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DATASET_PATH = DATA_DIR / "2cls_spam_text_cls.csv"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "error_analysis.json"

MODEL_NAME = "intfloat/multilingual-e5-base"

TEST_SIZE = 0.1
RANDOM_STATE = 42
BATCH_SIZE = 32
DEFAULT_K = 3

MESSAGE_COLUMN = "Message"
LABEL_COLUMN = "Category"