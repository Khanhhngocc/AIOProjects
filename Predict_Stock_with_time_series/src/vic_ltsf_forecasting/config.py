from pathlib import Path
import torch

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"

DATA_PATH = DATA_DIR / "VIC.csv"
GOOGLE_DRIVE_FILE_ID = "1QiDxOQM5h6RlJAPmjcr9ez4fLE4TNC2O"

# Data configuration
TARGET_COLUMN = "close_log"
LOOK_BACK_WINDOWS = [5, 30, 120, 480]
PREDICTION_LENGTH = 5
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15

# Training configuration
SEED = 59
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
MOVING_AVERAGE_WINDOW = 5
MODEL_NAMES = ["Linear", "DLinear", "NLinear"]

# Select the best available device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")