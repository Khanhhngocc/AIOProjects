import torch

from torch.utils.data import Dataset, DataLoader
from vic_ltsf_forecasting.config import (
    LOOK_BACK_WINDOWS,
    PREDICTION_LENGTH,
    BATCH_SIZE
)


class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len, pred_len, start, end, scaler):
        self.series = series
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.indices = list(range(start, end))
        self.scaler = scaler

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        start = self.indices[index]

        input_end = start + self.seq_len
        target_end = input_end + self.pred_len

        x = self.series[start:input_end]
        y = self.series[input_end:target_end]

        x = self.scaler.transform(x.reshape(-1, 1)).flatten()
        y = self.scaler.transform(y.reshape(-1, 1)).flatten()

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )


def create_loaders(series, scaler, train_end, valid_end):
    loaders = {}

    for look_back in LOOK_BACK_WINDOWS:
        train_stop = train_end - look_back - PREDICTION_LENGTH + 1

        valid_start = train_end - look_back
        valid_stop = valid_end - look_back - PREDICTION_LENGTH + 1

        test_start = valid_end - look_back
        test_stop = len(series) - look_back - PREDICTION_LENGTH + 1

        train_dataset = TimeSeriesDataset(
            series, look_back, PREDICTION_LENGTH,
            0, train_stop, scaler
        )

        valid_dataset = TimeSeriesDataset(
            series, look_back, PREDICTION_LENGTH,
            valid_start, valid_stop, scaler
        )

        test_dataset = TimeSeriesDataset(
            series, look_back, PREDICTION_LENGTH,
            test_start, test_stop, scaler
        )

        loaders[look_back] = {
            "train": DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True
            ),
            "valid": DataLoader(
                valid_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False
            ),
            "test": DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False
            )
        }

    return loaders