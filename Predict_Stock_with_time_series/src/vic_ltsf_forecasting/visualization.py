import pandas as pd
import matplotlib.pyplot as plt


def plot_rmse_comparison(summary, save_path):
    plt.figure(figsize=(10, 6))

    for model_name in summary["model"].unique():
        model_data = summary[
            summary["model"] == model_name
        ].sort_values("look_back")

        plt.plot(
            model_data["look_back"],
            model_data["rmse"],
            marker="o",
            label=model_name
        )

    plt.xlabel("Look-back Days")
    plt.ylabel("RMSE (VND)")
    plt.title("Model RMSE Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_loss(history, model_name, look_back, save_path):
    plt.figure(figsize=(8, 5))

    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        history["epoch"],
        history["valid_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Normalized MSE")
    plt.title(
        f"{model_name} - "
        f"{look_back}-Day Look-back"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_test_predictions(
    predictions,
    model_name,
    look_back,
    save_path
):
    day_5 = predictions[
        predictions["horizon_day"] == 5
    ]

    plt.figure(figsize=(12, 5))

    plt.plot(
        day_5["actual"].to_numpy(),
        label="Actual"
    )

    plt.plot(
        day_5["predicted"].to_numpy(),
        label="Predicted"
    )

    plt.xlabel("Test Window")
    plt.ylabel("VIC Closing Price (VND)")
    plt.title(
        f"Day-5 Forecast - {model_name}, "
        f"{look_back}-Day Look-back"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_future_forecast(dataframe, forecast_dates, forecast_prices, model_name, look_back, save_path):
    recent_history = dataframe.tail(120)

    plt.figure(figsize=(12, 5))

    plt.plot(
        recent_history["time"],
        recent_history["close"],
        label="Historical VIC Close"
    )

    plt.plot(
        forecast_dates,
        forecast_prices,
        marker="o",
        linestyle="--",
        label=f"Forecast: {model_name} ({look_back}-day look-back)"
    )

    plt.axvline(
        dataframe["time"].iloc[-1],
        linestyle=":",
        label="Forecast Start"
    )

    plt.xlabel("Date")
    plt.ylabel("VIC Closing Price")
    plt.title("VIC Closing Price: Next 5-Day Forecast")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()