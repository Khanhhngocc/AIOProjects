import numpy as np
import pandas as pd
import torch

@torch.no_grad()
def forecast_next_days(
    dataframe,
    model,
    scaler,
    look_back,
    pred_len,
    device,
):
    """Forecast future VIC closing prices from the latest observations."""
    model.eval()

    latest_log_prices = dataframe["close_log"].iloc[
        -look_back:
    ].to_numpy(dtype=np.float32)

    latest_input_scaled = scaler.transform(
        latest_log_prices.reshape(-1, 1)
    ).flatten()

    input_tensor = torch.tensor(
        latest_input_scaled,
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    prediction_scaled = model(
        input_tensor
    ).cpu().numpy().flatten()

    prediction_log = scaler.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    ).flatten()

    prediction_price = np.exp(prediction_log)

    latest_date = dataframe["time"].iloc[-1]

    # Use business days for future stock-market dates
    forecast_dates = pd.bdate_range(
        start=latest_date + pd.offsets.BDay(1),
        periods=pred_len,
    )

    return forecast_dates, prediction_price

def create_forecast_dataframe(forecast_dates, forecast_prices):
    """Create a readable forecast table."""
    return pd.DataFrame(
        {
            "Forecast Date": forecast_dates,
            "Predicted VIC Close (VND)": forecast_prices,
        }
    )