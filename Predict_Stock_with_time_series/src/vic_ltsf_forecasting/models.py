import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    """Map historical values directly to future values."""

    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        return self.linear(x)

class NLinear(nn.Module):
    """Forecast relative changes after removing the final observed level."""

    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        last_value = x[:, -1:].detach()
        x_normalized = x - last_value
        prediction = self.linear(x_normalized)
        return prediction + last_value

class DLinear(nn.Module):
    """Decompose the input into trend and seasonal components."""

    def __init__(self, seq_len, pred_len, moving_avg=5):
        super().__init__()

        moving_avg = min(moving_avg, seq_len)
        if moving_avg % 2 == 0:
            moving_avg -= 1

        self.moving_avg = max(moving_avg, 1)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)

        kernel = torch.ones(1, 1, self.moving_avg) / self.moving_avg
        self.register_buffer("average_kernel", kernel)

    def decompose(self, x):
        """Split input into moving-average trend and residual seasonality."""
        x = x.unsqueeze(1)
        padding = self.moving_avg // 2

        x_padded = F.pad(
            x,
            pad=(padding, padding),
            mode="replicate",
        )

        trend = F.conv1d(
            x_padded,
            self.average_kernel,
        ).squeeze(1)

        seasonal = x.squeeze(1) - trend
        return trend, seasonal

    def forward(self, x):
        trend, seasonal = self.decompose(x)
        trend_prediction = self.linear_trend(trend)
        seasonal_prediction = self.linear_seasonal(seasonal)
        return trend_prediction + seasonal_prediction

def create_model(model_name, seq_len, pred_len, moving_avg=5):
    """Create a forecasting model from its name."""
    if model_name == "Linear":
        return Linear(seq_len, pred_len)

    if model_name == "NLinear":
        return NLinear(seq_len, pred_len)

    if model_name == "DLinear":
        return DLinear(
            seq_len=seq_len,
            pred_len=pred_len,
            moving_avg=moving_avg,
        )

    raise ValueError(f"Unknown model name: {model_name}")