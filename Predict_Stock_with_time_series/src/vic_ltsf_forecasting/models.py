import torch
import torch.nn as nn

from vic_ltsf_forecasting.config import MOVING_AVERAGE_WINDOW


class Linear(nn.Module):
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        return self.linear(x)


class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, moving_avg=5):
        super().__init__()

        moving_avg = min(moving_avg, seq_len)

        if moving_avg % 2 == 0:
            moving_avg -= 1

        self.moving_avg = max(moving_avg, 1)

        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)

        average_kernel = (
            torch.ones(1, 1, self.moving_avg)
            / self.moving_avg
        )

        self.register_buffer(
            "average_kernel",
            average_kernel
        )

    def decompose(self, x):
        x_reshaped = x.unsqueeze(1)
        padding = self.moving_avg // 2

        x_padded = torch.nn.functional.pad(
            x_reshaped,
            (padding, padding),
            mode="replicate"
        )

        trend = torch.nn.functional.conv1d(
            x_padded,
            self.average_kernel
        ).squeeze(1)

        seasonal = x - trend

        return trend, seasonal

    def forward(self, x):
        trend, seasonal = self.decompose(x)

        return (
            self.linear_trend(trend)
            + self.linear_seasonal(seasonal)
        )


class NLinear(nn.Module):
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        last_value = x[:, -1:].detach()
        x = x - last_value

        prediction = self.linear(x)

        return prediction + last_value


def create_model(model_name, seq_len, pred_len):
    if model_name == "Linear":
        return Linear(seq_len, pred_len)

    if model_name == "DLinear":
        return DLinear(
            seq_len,
            pred_len,
            MOVING_AVERAGE_WINDOW
        )

    if model_name == "NLinear":
        return NLinear(seq_len, pred_len)

    raise ValueError(f"Unknown model: {model_name}")