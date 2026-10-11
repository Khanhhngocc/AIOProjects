import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def forecast_next_days(dataframe, model, scaler, look_back, pred_len, device):
    model.eval()

    latest_values = dataframe["close_log"].iloc[-look_back:].to_numpy(dtype=np.float32)
    normalized_values = scaler.transform(latest_values.reshape(-1, 1)).flatten()

    input_tensor = torch.tensor(
        normalized_values,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    prediction_normalized = model(input_tensor).cpu().numpy().flatten()

    prediction_log = scaler.inverse_transform(
        prediction_normalized.reshape(-1, 1)
    ).flatten()

    prediction_price = np.exp(prediction_log)

    latest_date = dataframe["time"].iloc[-1]

    forecast_dates = pd.bdate_range(
        start=latest_date + pd.offsets.BDay(1),
        periods=pred_len
    )

    return forecast_dates, prediction_price