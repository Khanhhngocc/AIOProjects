import pandas as pd
import torch

from vic_ltsf_forecasting.config import (
    RESULT_DIR,
    MODEL_DIR,
    PREDICTION_LENGTH,
    DEVICE
)
from vic_ltsf_forecasting.data_loader import load_dataframe, prepare_series
from vic_ltsf_forecasting.models import create_model
from vic_ltsf_forecasting.forecaster import forecast_next_days
from vic_ltsf_forecasting.visualization import plot_future_forecast


RUN_NAME = "run_01"


def main():
    # 1. Load best experiment
    run_dir = RESULT_DIR / RUN_NAME
    summary = pd.read_csv(run_dir / "summary.csv")

    best = summary.loc[summary["rmse"].idxmin()]
    model_name = best["model"]
    look_back = int(best["look_back"])

    print("Best experiment:")
    print(f"Model: {model_name}")
    print(f"Look-back: {look_back} days")
    print(f"RMSE: {best['rmse']:,.2f}")
    print(f"MAE: {best['mae']:,.2f}")
    print(f"R²: {best['r2']:.4f}")

    # 2. Load data and scaler
    dataframe = load_dataframe()
    _, scaler, _, _ = prepare_series(dataframe)

    # 3. Load trained model
    model = create_model(model_name, look_back, PREDICTION_LENGTH)

    checkpoint_path = (
        MODEL_DIR /
        RUN_NAME /
        f"vic_{model_name.lower()}_{look_back}d_best.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)

    # 4. Forecast next 5 trading days
    forecast_dates, forecast_prices = forecast_next_days(
        dataframe,
        model,
        scaler,
        look_back,
        PREDICTION_LENGTH,
        DEVICE
    )

    forecast_df = pd.DataFrame({
        "forecast_date": forecast_dates,
        "predicted_close": forecast_prices
    })

    print("\nNext 5-day VIC forecast:")
    print(forecast_df.to_string(index=False))

    # 5. Save results
    forecast_df.to_csv(
        run_dir / "future_forecast.csv",
        index=False
    )

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_future_forecast(
        dataframe,
        forecast_dates,
        forecast_prices,
        model_name,
        look_back,
        figures_dir / "future_forecast.png"
    )

    print(f"\nForecast saved at: {run_dir / 'future_forecast.csv'}")
    print(f"Figure saved at: {figures_dir / 'future_forecast.png'}")


if __name__ == "__main__":
    main()