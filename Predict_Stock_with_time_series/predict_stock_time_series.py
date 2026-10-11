# =========================================================
# 1. PROBLEM DEFINITION
#    Forecast the next 5 VIC closing prices using the past
#    5, 30, 120, or 480 trading days.
#
#    Models:
#    - Linear
#    - NLinear
#    - DLinear
# =========================================================


# =========================================================
# 2. IMPORT LIBRARIES
# =========================================================
from pathlib import Path
import random

import gdown
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# =========================================================
# 3. CONFIGURATION
# =========================================================
SEED = 59

DATA_PATH = Path("VIC.csv")
GOOGLE_DRIVE_FILE_ID = "1QiDxOQM5h6RlJAPmjcr9ez4fLE4TNC2O"

TARGET_COLUMN = "close_log"
LOOK_BACK_WINDOWS = [5, 30, 120, 480]
PREDICTION_LENGTH = 5

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
MOVING_AVERAGE_WINDOW = 5

MODEL_DIR = Path("./model")
RESULT_DIR = Path("./results")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# =========================================================
# 4. REPRODUCIBILITY
# =========================================================
def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

print(f"Using device: {DEVICE}")


# =========================================================
# 5. DOWNLOAD AND LOAD DATA
# =========================================================
def download_data_if_needed(data_path: Path, file_id: str) -> None:
    """Download the dataset from Google Drive when it is missing."""
    if not data_path.exists():
        print("Downloading VIC.csv...")
        gdown.download(
            id=file_id,
            output=str(data_path),
            quiet=False,
        )


download_data_if_needed(DATA_PATH, GOOGLE_DRIVE_FILE_ID)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")
print("\nOriginal columns:")
print(df.columns.tolist())


# =========================================================
# 6. DATA CLEANING AND FEATURE CREATION
# =========================================================
def prepare_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data and create time-series features."""
    dataframe = dataframe.copy()

    dataframe["time"] = pd.to_datetime(dataframe["time"])
    dataframe["close"] = pd.to_numeric(dataframe["close"], errors="coerce")

    if "volume" in dataframe.columns:
        dataframe["volume"] = pd.to_numeric(
            dataframe["volume"],
            errors="coerce",
        )

    dataframe = dataframe.sort_values("time").reset_index(drop=True)

    # A log transformation requires strictly positive values.
    dataframe = dataframe.dropna(subset=["time", "close"])
    dataframe = dataframe[dataframe["close"] > 0].reset_index(drop=True)

    # Create additional features for EDA.
    dataframe["daily_return"] = dataframe["close"].pct_change().fillna(0.0)
    dataframe["close_log"] = np.log(dataframe["close"])

    return dataframe


df = prepare_dataframe(df)

print(f"\nDate range: {df['time'].min().date()} to {df['time'].max().date()}")
print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df[["close", "close_log", "daily_return"]].describe())


# =========================================================
# 7. EXPLORATORY DATA ANALYSIS
# =========================================================
def plot_eda(dataframe: pd.DataFrame) -> None:
    """Visualize stock price, log price, returns, and trading volume."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "VIC Stock Data Analysis",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Closing price over time.
    axes[0, 0].plot(
        dataframe["time"],
        dataframe["close"],
        linewidth=1.5,
        color="blue",
        alpha=0.8,
    )
    axes[0, 0].set_title("VIC Closing Price Over Time")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Price (VND)")
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Log-transformed closing price.
    axes[0, 1].plot(
        dataframe["time"],
        dataframe["close_log"],
        linewidth=1.5,
        color="green",
        alpha=0.8,
    )
    axes[0, 1].set_title("Log-Transformed Closing Price")
    axes[0, 1].set_xlabel("Date")
    axes[0, 1].set_ylabel("Log Price")
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: Daily returns and the 20-day rolling average.
    returns_dataframe = dataframe[["time", "daily_return"]].copy()
    returns_dataframe["return_ma20"] = (
        returns_dataframe["daily_return"].rolling(20).mean()
    )

    axes[1, 0].plot(
        returns_dataframe["time"],
        returns_dataframe["daily_return"],
        linewidth=1.0,
        alpha=0.7,
        label="Daily Return",
    )
    axes[1, 0].plot(
        returns_dataframe["time"],
        returns_dataframe["return_ma20"],
        linewidth=2.0,
        linestyle="--",
        label="20-Day Average",
    )
    axes[1, 0].axhline(0, color="black", linewidth=1, alpha=0.6)
    axes[1, 0].set_title("Daily Returns Over Time")
    axes[1, 0].set_xlabel("Date")
    axes[1, 0].set_ylabel("Return")
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    # Plot 4: Trading volume, if the column exists.
    if "volume" in dataframe.columns:
        axes[1, 1].plot(
            dataframe["time"],
            dataframe["volume"],
            linewidth=1.2,
            color="purple",
            alpha=0.8,
        )
        axes[1, 1].set_title("Trading Volume Over Time")
        axes[1, 1].set_xlabel("Date")
        axes[1, 1].set_ylabel("Volume")
        axes[1, 1].grid(alpha=0.3)
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Volume column is unavailable.",
            ha="center",
            va="center",
        )
        axes[1, 1].axis("off")

    for axis in axes.flat:
        if axis.axison:
            axis.xaxis.set_major_locator(mdates.AutoDateLocator())
            axis.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(
                    mdates.AutoDateLocator()
                )
            )

    plt.tight_layout()
    plt.show()


plot_eda(df)


# =========================================================
# 8. TIME-BASED DATA SPLIT
# =========================================================
def get_time_split_indices(
    total_points: int,
    train_ratio: float,
    valid_ratio: float,
):
    """
    Create chronological split boundaries on raw time points.

    Train : [0, train_end)
    Valid : [train_end, valid_end)
    Test  : [valid_end, total_points)
    """
    train_end = int(total_points * train_ratio)
    valid_end = train_end + int(total_points * valid_ratio)

    return train_end, valid_end


series = df[TARGET_COLUMN].to_numpy(dtype=np.float32)

TRAIN_END, VALID_END = get_time_split_indices(
    total_points=len(series),
    train_ratio=TRAIN_RATIO,
    valid_ratio=VALID_RATIO,
)

print("\nTime-based split:")
print(f"Train: index 0 to {TRAIN_END - 1}")
print(f"Valid: index {TRAIN_END} to {VALID_END - 1}")
print(f"Test : index {VALID_END} to {len(series) - 1}")

print(f"\nTrain end date: {df['time'].iloc[TRAIN_END - 1].date()}")
print(f"Valid end date: {df['time'].iloc[VALID_END - 1].date()}")
print(f"Test end date : {df['time'].iloc[-1].date()}")


# =========================================================
# 9. TRAIN-ONLY NORMALIZATION
# =========================================================
# The scaler is fitted only on training values.
# This prevents information leakage from validation and test data.
scaler = StandardScaler()

train_values = series[:TRAIN_END].reshape(-1, 1)
scaler.fit(train_values)


# =========================================================
# 10. TIME SERIES DATASET
# =========================================================
class UnivariateTimeSeriesDataset(Dataset):
    """
    Create sliding windows for direct multi-step forecasting.

    Input:
        x = historical values with length seq_len

    Target:
        y = future values with length pred_len
    """

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
        self.start_window_index = start_window_index
        self.end_window_index = end_window_index
        self.scaler = scaler

        # Window start indices are stored explicitly.
        self.window_indices = list(
            range(start_window_index, end_window_index)
        )

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, index):
        window_start = self.window_indices[index]

        input_end = window_start + self.seq_len
        target_end = input_end + self.pred_len

        x = self.series[window_start:input_end]
        y = self.series[input_end:target_end]

        x_normalized = self.scaler.transform(
            x.reshape(-1, 1)
        ).flatten()

        y_normalized = self.scaler.transform(
            y.reshape(-1, 1)
        ).flatten()

        return (
            torch.tensor(x_normalized, dtype=torch.float32),
            torch.tensor(y_normalized, dtype=torch.float32),
        )

    def get_target_dates(self, index):
        """Return the actual dates of the forecast horizon."""
        window_start = self.window_indices[index]
        target_start = window_start + self.seq_len
        target_end = target_start + self.pred_len

        return self.dates[target_start:target_end]


# =========================================================
# 11. DATASET CREATION FOR EACH LOOK-BACK WINDOW
# =========================================================
def create_datasets_for_window(
    series,
    dates,
    seq_len,
    pred_len,
    train_end,
    valid_end,
    scaler,
):
    """
    Create train, validation, and test datasets.

    Validation and test inputs may use earlier historical values,
    but their target values belong only to their own time segment.
    """
    total_points = len(series)

    # All training targets must end before train_end.
    train_start = 0
    train_stop = train_end - seq_len - pred_len + 1

    # Validation targets start exactly at train_end.
    valid_start = train_end - seq_len
    valid_stop = valid_end - seq_len - pred_len + 1

    # Test targets start exactly at valid_end.
    test_start = valid_end - seq_len
    test_stop = total_points - seq_len - pred_len + 1

    if train_stop <= train_start:
        raise ValueError(
            f"seq_len={seq_len} is too large for the training split."
        )

    train_dataset = UnivariateTimeSeriesDataset(
        series=series,
        dates=dates,
        seq_len=seq_len,
        pred_len=pred_len,
        start_window_index=train_start,
        end_window_index=train_stop,
        scaler=scaler,
    )

    valid_dataset = UnivariateTimeSeriesDataset(
        series=series,
        dates=dates,
        seq_len=seq_len,
        pred_len=pred_len,
        start_window_index=valid_start,
        end_window_index=valid_stop,
        scaler=scaler,
    )

    test_dataset = UnivariateTimeSeriesDataset(
        series=series,
        dates=dates,
        seq_len=seq_len,
        pred_len=pred_len,
        start_window_index=test_start,
        end_window_index=test_stop,
        scaler=scaler,
    )

    return train_dataset, valid_dataset, test_dataset


datasets_by_window = {}

for look_back in LOOK_BACK_WINDOWS:
    train_dataset, valid_dataset, test_dataset = (
        create_datasets_for_window(
            series=series,
            dates=df["time"].to_numpy(),
            seq_len=look_back,
            pred_len=PREDICTION_LENGTH,
            train_end=TRAIN_END,
            valid_end=VALID_END,
            scaler=scaler,
        )
    )

    datasets_by_window[look_back] = {
        "train": train_dataset,
        "valid": valid_dataset,
        "test": test_dataset,
    }

    print(
        f"\nLook-back = {look_back} days"
        f"\n- Train samples: {len(train_dataset)}"
        f"\n- Valid samples: {len(valid_dataset)}"
        f"\n- Test samples : {len(test_dataset)}"
    )


# =========================================================
# 12. DATALOADERS
# =========================================================
loaders_by_window = {}

for look_back, datasets in datasets_by_window.items():
    loaders_by_window[look_back] = {
        "train": DataLoader(
            datasets["train"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        ),
        "valid": DataLoader(
            datasets["valid"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False,
        ),
    }

sample_inputs, sample_targets = next(
    iter(loaders_by_window[LOOK_BACK_WINDOWS[0]]["train"])
)

print("\nBatch shapes:")
print(f"Inputs : {sample_inputs.shape}")
print(f"Targets: {sample_targets.shape}")


# =========================================================
# 13. LINEAR MODEL
# =========================================================
class Linear(nn.Module):
    """
    Map seq_len historical values directly to pred_len future values.
    """

    def __init__(self, seq_len, pred_len):
        super().__init__()

        self.linear = nn.Linear(
            in_features=seq_len,
            out_features=pred_len,
        )

    def forward(self, x):
        return self.linear(x)


# =========================================================
# 14. DLINEAR MODEL
# =========================================================
class DLinear(nn.Module):
    """
    Decompose input into trend and seasonal components.

    Each component is forecasted by a separate linear layer.
    """

    def __init__(self, seq_len, pred_len, moving_avg=5):
        super().__init__()

        # An odd window keeps the moving-average output length unchanged.
        moving_avg = min(moving_avg, seq_len)

        if moving_avg % 2 == 0:
            moving_avg -= 1

        self.moving_avg = max(moving_avg, 1)

        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)

        average_kernel = torch.ones(
            1,
            1,
            self.moving_avg,
        ) / self.moving_avg

        # register_buffer moves this tensor with the model device.
        self.register_buffer("average_kernel", average_kernel)

    def decompose(self, x):
        """Split a sequence into trend and seasonal components."""
        x_reshaped = x.unsqueeze(1)

        padding = self.moving_avg // 2

        x_padded = torch.nn.functional.pad(
            x_reshaped,
            pad=(padding, padding),
            mode="replicate",
        )

        trend = torch.nn.functional.conv1d(
            x_padded,
            self.average_kernel,
        ).squeeze(1)

        seasonal = x - trend

        return trend, seasonal

    def forward(self, x):
        trend, seasonal = self.decompose(x)

        trend_prediction = self.linear_trend(trend)
        seasonal_prediction = self.linear_seasonal(seasonal)

        return trend_prediction + seasonal_prediction


# =========================================================
# 15. NLINEAR MODEL
# =========================================================
class NLinear(nn.Module):
    """
    Re-center each input window using its final observed value.

    The linear layer learns relative changes instead of absolute levels.
    """

    def __init__(self, seq_len, pred_len):
        super().__init__()

        self.linear = nn.Linear(
            in_features=seq_len,
            out_features=pred_len,
        )

    def forward(self, x):
        last_value = x[:, -1:].detach()

        x_normalized = x - last_value
        prediction_normalized = self.linear(x_normalized)

        prediction = prediction_normalized + last_value

        return prediction


# =========================================================
# 16. TRAINING FUNCTION
# =========================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train a model for one epoch."""
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)

        loss.backward()
        optimizer.step()

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


# =========================================================
# 17. VALIDATION FUNCTION
# =========================================================
@torch.no_grad()
def evaluate_loss(model, dataloader, criterion, device):
    """Calculate average normalized MSE loss."""
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


# =========================================================
# 18. MODEL TRAINING AND CHECKPOINT SAVING
# =========================================================
def train_model(
    model,
    train_loader,
    valid_loader,
    model_name,
    look_back,
    device,
    num_epochs,
    learning_rate,
):
    """
    Train a model and save its best validation-loss checkpoint.
    """
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.5,
    )

    model = model.to(device)

    history = {
        "train_loss": [],
        "valid_loss": [],
    }

    best_valid_loss = float("inf")

    checkpoint_path = (
        MODEL_DIR / f"vic_{model_name.lower()}_{look_back}d_best.pt"
    )

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        valid_loss = evaluate_loss(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "look_back": look_back,
                    "pred_len": PREDICTION_LENGTH,
                    "best_valid_loss": best_valid_loss,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

        print(
            f"{model_name:7s} | "
            f"Look-back: {look_back:3d} | "
            f"Epoch: {epoch:02d}/{num_epochs} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Valid MSE: {valid_loss:.6f}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    return model, history, checkpoint_path


# =========================================================
# 19. TEST EVALUATION
# =========================================================
@torch.no_grad()
def evaluate_predictions(model, dataloader, scaler, device):
    """
    Calculate metrics on the original VIC price scale.

    The model predicts normalized log prices.
    They are converted back to actual VND closing prices here.
    """
    model.eval()

    all_predictions_normalized = []
    all_targets_normalized = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)

        predictions = model(batch_x)

        all_predictions_normalized.append(
            predictions.cpu().numpy()
        )
        all_targets_normalized.append(
            batch_y.numpy()
        )

    predictions_normalized = np.concatenate(
        all_predictions_normalized,
        axis=0,
    )

    targets_normalized = np.concatenate(
        all_targets_normalized,
        axis=0,
    )

    predictions_log = scaler.inverse_transform(
        predictions_normalized.reshape(-1, 1)
    ).reshape(predictions_normalized.shape)

    targets_log = scaler.inverse_transform(
        targets_normalized.reshape(-1, 1)
    ).reshape(targets_normalized.shape)

    # The target was log(close), so exponentiate to recover price.
    predictions_price = np.exp(predictions_log)
    targets_price = np.exp(targets_log)

    predictions_flat = predictions_price.flatten()
    targets_flat = targets_price.flatten()

    mse = mean_squared_error(targets_flat, predictions_flat)
    mae = mean_absolute_error(targets_flat, predictions_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_flat, predictions_flat)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": predictions_price,
        "actuals": targets_price,
    }


# =========================================================
# 20. CREATE ALL MODEL INSTANCES
# =========================================================
def create_model(model_name, seq_len, pred_len):
    """Create one LTSF-Linear model."""
    if model_name == "Linear":
        return Linear(seq_len, pred_len)

    if model_name == "DLinear":
        return DLinear(
            seq_len=seq_len,
            pred_len=pred_len,
            moving_avg=MOVING_AVERAGE_WINDOW,
        )

    if model_name == "NLinear":
        return NLinear(seq_len, pred_len)

    raise ValueError(f"Unknown model name: {model_name}")


MODEL_NAMES = ["Linear", "DLinear", "NLinear"]

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 21. TRAIN ALL EXPERIMENTS
# =========================================================
trained_models = {}
loss_histories = {}
test_results = {}

for look_back in LOOK_BACK_WINDOWS:
    print(f"\n{'=' * 75}")
    print(
        f"Training models with {look_back} days "
        f"of history -> {PREDICTION_LENGTH} future days"
    )
    print(f"{'=' * 75}")

    trained_models[look_back] = {}
    loss_histories[look_back] = {}
    test_results[look_back] = {}

    train_loader = loaders_by_window[look_back]["train"]
    valid_loader = loaders_by_window[look_back]["valid"]
    test_loader = loaders_by_window[look_back]["test"]

    for model_name in MODEL_NAMES:
        set_seed(SEED)

        model = create_model(
            model_name=model_name,
            seq_len=look_back,
            pred_len=PREDICTION_LENGTH,
        )

        trained_model, history, checkpoint_path = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            model_name=model_name,
            look_back=look_back,
            device=DEVICE,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
        )

        result = evaluate_predictions(
            model=trained_model,
            dataloader=test_loader,
            scaler=scaler,
            device=DEVICE,
        )

        trained_models[look_back][model_name] = trained_model
        loss_histories[look_back][model_name] = history
        test_results[look_back][model_name] = result

        print(
            f"Test result -> {model_name} | "
            f"RMSE: {result['rmse']:,.0f} VND | "
            f"MAE: {result['mae']:,.0f} VND | "
            f"R²: {result['r2']:.4f}"
        )


# =========================================================
# 22. TRAINING HISTORY VISUALIZATION
# =========================================================
def plot_loss_curves(loss_histories, look_back_windows):
    """Plot train and validation loss for every model."""
    for look_back in look_back_windows:
        plt.figure(figsize=(10, 5))

        for model_name in MODEL_NAMES:
            history = loss_histories[look_back][model_name]

            plt.plot(
                history["train_loss"],
                label=f"{model_name} - Train",
            )

            plt.plot(
                history["valid_loss"],
                linestyle="--",
                label=f"{model_name} - Valid",
            )

        plt.title(
            f"Training History: {look_back}-Day Input "
            f"-> {PREDICTION_LENGTH}-Day Forecast"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Normalized MSE Loss")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


plot_loss_curves(
    loss_histories=loss_histories,
    look_back_windows=LOOK_BACK_WINDOWS,
)


# =========================================================
# 23. TEST PERFORMANCE TABLE
# =========================================================
summary_rows = []

for look_back in LOOK_BACK_WINDOWS:
    for model_name in MODEL_NAMES:
        result = test_results[look_back][model_name]

        summary_rows.append(
            {
                "Look-back Days": look_back,
                "Model": model_name,
                "RMSE (VND)": result["rmse"],
                "MAE (VND)": result["mae"],
                "R²": result["r2"],
            }
        )

summary_df = pd.DataFrame(summary_rows).sort_values(
    by=["Look-back Days", "RMSE (VND)"]
).reset_index(drop=True)

print("\nTest Performance Summary:")
print(summary_df.to_string(index=False))

summary_df.to_csv(
    RESULT_DIR / "vic_ltsf_performance_summary.csv",
    index=False,
)


# =========================================================
# 24. PREDICTION VISUALIZATION
# =========================================================
def plot_test_predictions(test_results, look_back_windows):
    """
    Plot the fifth-day prediction of each forecast window.

    This lets us compare the same horizon step across models.
    """
    figure, axes = plt.subplots(2, 2, figsize=(18, 12))
    figure.suptitle(
        "VIC Test Predictions: Day-5 Forecast",
        fontsize=18,
        fontweight="bold",
    )

    colors = {
        "Linear": "red",
        "DLinear": "green",
        "NLinear": "blue",
    }

    for index, look_back in enumerate(look_back_windows):
        axis = axes[index // 2, index % 2]

        actual_values = test_results[look_back]["Linear"]["actuals"][:, -1]

        axis.plot(
            actual_values,
            color="black",
            linewidth=2,
            label="Actual",
            alpha=0.8,
        )

        for model_name in MODEL_NAMES:
            result = test_results[look_back][model_name]
            prediction_values = result["predictions"][:, -1]

            axis.plot(
                prediction_values,
                color=colors[model_name],
                linewidth=1.5,
                linestyle="--",
                label=(
                    f"{model_name} "
                    f"(RMSE: {result['rmse']:,.0f})"
                ),
                alpha=0.8,
            )

        best_model = min(
            MODEL_NAMES,
            key=lambda name: test_results[look_back][name]["rmse"],
        )

        axis.set_title(
            f"{look_back}-Day Input -> "
            f"{PREDICTION_LENGTH}-Day Forecast"
        )
        axis.set_xlabel("Test Window")
        axis.set_ylabel("VIC Closing Price (VND)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=9)

        axis.text(
            0.02,
            0.97,
            f"Best model: {best_model}",
            transform=axis.transAxes,
            verticalalignment="top",
            bbox={
                "boxstyle": "round",
                "facecolor": "wheat",
                "alpha": 0.8,
            },
        )

    plt.tight_layout()
    plt.show()


plot_test_predictions(
    test_results=test_results,
    look_back_windows=LOOK_BACK_WINDOWS,
)


# =========================================================
# 25. INFERENCE FUNCTION
# =========================================================
@torch.no_grad()
def forecast_next_days(
    dataframe,
    model,
    scaler,
    look_back,
    pred_len,
    device,
):
    """
    Forecast future closing prices from the latest available history.

    Returns:
        forecast_dates
        forecast_prices
    """
    model.eval()

    latest_log_prices = dataframe["close_log"].iloc[-look_back:].to_numpy(
        dtype=np.float32
    )

    latest_input_normalized = scaler.transform(
        latest_log_prices.reshape(-1, 1)
    ).flatten()

    input_tensor = torch.tensor(
        latest_input_normalized,
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    prediction_normalized = model(input_tensor).cpu().numpy().flatten()

    prediction_log = scaler.inverse_transform(
        prediction_normalized.reshape(-1, 1)
    ).flatten()

    prediction_price = np.exp(prediction_log)

    latest_date = dataframe["time"].iloc[-1]

    # BDay creates business-day dates for stock forecasts.
    forecast_dates = pd.bdate_range(
        start=latest_date + pd.offsets.BDay(1),
        periods=pred_len,
    )

    return forecast_dates, prediction_price


# =========================================================
# 26. FORECAST THE NEXT 5 DAYS
# =========================================================
best_row = summary_df.iloc[0]

best_look_back = int(best_row["Look-back Days"])
best_model_name = best_row["Model"]
best_model = trained_models[best_look_back][best_model_name]

forecast_dates, forecast_prices = forecast_next_days(
    dataframe=df,
    model=best_model,
    scaler=scaler,
    look_back=best_look_back,
    pred_len=PREDICTION_LENGTH,
    device=DEVICE,
)

forecast_df = pd.DataFrame(
    {
        "Forecast Date": forecast_dates,
        "Predicted VIC Close (VND)": forecast_prices,
    }
)

print("\nBest experiment:")
print(f"Model     : {best_model_name}")
print(f"Look-back : {best_look_back} days")

print("\nNext 5-day VIC forecast:")
print(forecast_df.to_string(index=False))

forecast_df.to_csv(
    RESULT_DIR / "vic_next_5_day_forecast.csv",
    index=False,
)


# =========================================================
# 27. FORECAST VISUALIZATION
# =========================================================
recent_days_to_show = 120

recent_history = df.tail(recent_days_to_show)

plt.figure(figsize=(14, 6))

plt.plot(
    recent_history["time"],
    recent_history["close"],
    label="Historical VIC Close",
    linewidth=2,
)

plt.plot(
    forecast_dates,
    forecast_prices,
    marker="o",
    linestyle="--",
    linewidth=2,
    label=(
        f"Forecast: {best_model_name} "
        f"({best_look_back}-day look-back)"
    ),
)

plt.axvline(
    df["time"].iloc[-1],
    color="black",
    linestyle=":",
    alpha=0.7,
    label="Forecast Start",
)

plt.title("VIC Closing Price: Historical Data and Next 5-Day Forecast")
plt.xlabel("Date")
plt.ylabel("VIC Closing Price (VND)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()