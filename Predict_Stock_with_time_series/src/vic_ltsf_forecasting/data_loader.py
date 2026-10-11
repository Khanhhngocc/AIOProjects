import gdown
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from vic_ltsf_forecasting.config import (
    DATA_PATH,
    GOOGLE_DRIVE_FILE_ID,
    TARGET_COLUMN,
    TRAIN_RATIO,
    VALID_RATIO
)


def download_data():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        print("Downloading VIC.csv...")
        gdown.download(
            id=GOOGLE_DRIVE_FILE_ID,
            output=str(DATA_PATH),
            quiet=False
        )


def load_dataframe():
    download_data()

    dataframe = pd.read_csv(DATA_PATH)
    dataframe["time"] = pd.to_datetime(dataframe["time"])
    dataframe["close"] = pd.to_numeric(dataframe["close"], errors="coerce")

    if "volume" in dataframe.columns:
        dataframe["volume"] = pd.to_numeric(
            dataframe["volume"],
            errors="coerce"
        )

    dataframe = dataframe.sort_values("time").reset_index(drop=True)
    dataframe = dataframe.dropna(subset=["time", "close"])
    dataframe = dataframe[dataframe["close"] > 0].reset_index(drop=True)

    dataframe["daily_return"] = dataframe["close"].pct_change().fillna(0.0)
    dataframe["close_log"] = np.log(dataframe["close"])

    return dataframe


def prepare_series(dataframe):
    series = dataframe[TARGET_COLUMN].to_numpy(dtype=np.float32)

    train_end = int(len(series) * TRAIN_RATIO)
    valid_end = train_end + int(len(series) * VALID_RATIO)

    scaler = StandardScaler()
    scaler.fit(series[:train_end].reshape(-1, 1))

    return series, scaler, train_end, valid_end