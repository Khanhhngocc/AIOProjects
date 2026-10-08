import torch
from torch.utils.data import Dataset, DataLoader

class UnivariateTimeSeriesDataset(Dataset):
    """Create sliding windows for direct multi-step forecasting."""

    def __init__(
        self,
        series,
        dates,
        seq_len,
        pred_len,
        start_window_index,
        end_window_index,
        scaler,
    ):
        self.series = series
        self.dates = dates
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler = scaler
        self.window_indices = list(range(start_window_index, end_window_index))

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, index):
        window_start = self.window_indices[index]
        input_end = window_start + self.seq_len
        target_end = input_end + self.pred_len

        x = self.series[window_start:input_end]
        y = self.series[input_end:target_end]

        x_scaled = self.scaler.transform(x.reshape(-1, 1)).flatten()
        y_scaled = self.scaler.transform(y.reshape(-1, 1)).flatten()

        return (
            torch.tensor(x_scaled, dtype=torch.float32),
            torch.tensor(y_scaled, dtype=torch.float32),
        )

    def get_target_dates(self, index):
        """Return actual dates corresponding to the prediction horizon."""
        window_start = self.window_indices[index]
        target_start = window_start + self.seq_len
        target_end = target_start + self.pred_len
        return self.dates[target_start:target_end]

def create_datasets_for_window(
    series,
    dates,
    seq_len,
    pred_len,
    train_end,
    valid_end,
    scaler,
):
    """Create train, validation, and test datasets for one look-back window."""
    total_points = len(series)

    # Training targets must remain completely inside the training segment
    train_start = 0
    train_stop = train_end - seq_len - pred_len + 1

    # Validation targets start at the validation boundary
    valid_start = train_end - seq_len
    valid_stop = valid_end - seq_len - pred_len + 1

    # Test targets start at the test boundary
    test_start = valid_end - seq_len
    test_stop = total_points - seq_len - pred_len + 1

    if train_stop <= train_start:
        raise ValueError(f"seq_len={seq_len} is too large for the training split.")

    train_dataset = UnivariateTimeSeriesDataset(
        series, dates, seq_len, pred_len,
        train_start, train_stop, scaler,
    )
    valid_dataset = UnivariateTimeSeriesDataset(
        series, dates, seq_len, pred_len,
        valid_start, valid_stop, scaler,
    )
    test_dataset = UnivariateTimeSeriesDataset(
        series, dates, seq_len, pred_len,
        test_start, test_stop, scaler,
    )

    return train_dataset, valid_dataset, test_dataset

def create_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size):
    """Create DataLoaders for training, validation, and testing."""
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        ),
        "valid": DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        ),
    }

def build_datasets_and_loaders(
    series,
    dates,
    look_back_windows,
    pred_len,
    train_end,
    valid_end,
    scaler,
    batch_size,
):
    """Build datasets and DataLoaders for every look-back configuration."""
    datasets_by_window = {}
    loaders_by_window = {}

    for look_back in look_back_windows:
        train_dataset, valid_dataset, test_dataset = create_datasets_for_window(
            series=series,
            dates=dates,
            seq_len=look_back,
            pred_len=pred_len,
            train_end=train_end,
            valid_end=valid_end,
            scaler=scaler,
        )

        datasets_by_window[look_back] = {
            "train": train_dataset,
            "valid": valid_dataset,
            "test": test_dataset,
        }

        loaders_by_window[look_back] = create_dataloaders(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
        )

        print(
            f"\nLook-back: {look_back} days | "
            f"Train: {len(train_dataset)} | "
            f"Valid: {len(valid_dataset)} | "
            f"Test: {len(test_dataset)}"
        )

    return datasets_by_window, loaders_by_window