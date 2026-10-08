import gdown
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from vic_ltsf_forecasting.config import DATA_PATH, DATA_DIR, GOOGLE_DRIVE_FILE_ID

def download_data_if_needed(data_path=DATA_PATH, file_id=GOOGLE_DRIVE_FILE_ID) -> None:
    """Download VIC.csv from Google Drive when the file does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print("Downloading VIC.csv...")
        gdown.download(id=file_id, output=str(data_path), quiet=False)

def load_dataframe(data_path=DATA_PATH) -> pd.DataFrame:
    """Load the raw VIC dataset."""
    download_data_if_needed(data_path)
    return pd.read_csv(data_path)

def prepare_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data and create time-series features."""
    dataframe = dataframe.copy()
    dataframe["time"] = pd.to_datetime(dataframe["time"])
    dataframe["close"] = pd.to_numeric(dataframe["close"], errors="coerce")

    if "volume" in dataframe.columns:
        dataframe["volume"] = pd.to_numeric(dataframe["volume"], errors="coerce")

    dataframe = dataframe.sort_values("time").reset_index(drop=True)
    dataframe = dataframe.dropna(subset=["time", "close"])
    dataframe = dataframe[dataframe["close"] > 0].reset_index(drop=True)

    # Create features used for analysis and forecasting
    dataframe["daily_return"] = dataframe["close"].pct_change().fillna(0.0)
    dataframe["close_log"] = np.log(dataframe["close"])

    return dataframe

def get_time_split_indices(total_points: int, train_ratio: float, valid_ratio: float):
    """Create chronological train, validation, and test boundaries."""
    train_end = int(total_points * train_ratio)
    valid_end = train_end + int(total_points * valid_ratio)
    return train_end, valid_end

def prepare_series_and_scaler(dataframe, target_column, train_ratio, valid_ratio):
    """Create the target series, chronological split, and train-only scaler."""
    series = dataframe[target_column].to_numpy(dtype=np.float32)
    train_end, valid_end = get_time_split_indices(
        total_points=len(series),
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )

    # Fit only on training values to prevent data leakage
    scaler = StandardScaler()
    scaler.fit(series[:train_end].reshape(-1, 1))

    return series, train_end, valid_end, scaler