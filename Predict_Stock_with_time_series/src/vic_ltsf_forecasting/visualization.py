import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def plot_eda(dataframe):
    """Visualize closing price, log price, daily returns, and volume."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "VIC Stock Data Analysis",
        fontsize=16,
        fontweight="bold",
    )

    axes[0, 0].plot(
        dataframe["time"],
        dataframe["close"],
        linewidth=1.5,
    )
    axes[0, 0].set_title("VIC Closing Price Over Time")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Price (VND)")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(
        dataframe["time"],
        dataframe["close_log"],
        linewidth=1.5,
    )
    axes[0, 1].set_title("Log-Transformed Closing Price")
    axes[0, 1].set_xlabel("Date")
    axes[0, 1].set_ylabel("Log Price")
    axes[0, 1].grid(alpha=0.3)

    returns_df = dataframe[["time", "daily_return"]].copy()
    returns_df["return_ma20"] = (
        returns_df["daily_return"].rolling(20).mean()
    )

    axes[1, 0].plot(
        returns_df["time"],
        returns_df["daily_return"],
        linewidth=1.0,
        alpha=0.7,
        label="Daily Return",
    )
    axes[1, 0].plot(
        returns_df["time"],
        returns_df["return_ma20"],
        linewidth=2.0,
        linestyle="--",
        label="20-Day Average",
    )
    axes[1, 0].axhline(
        0,
        linewidth=1,
        alpha=0.6,
    )
    axes[1, 0].set_title("Daily Returns Over Time")
    axes[1, 0].set_xlabel("Date")
    axes[1, 0].set_ylabel("Return")
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    if "volume" in dataframe.columns:
        axes[1, 1].plot(
            dataframe["time"],
            dataframe["volume"],
            linewidth=1.2,
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
            locator = mdates.AutoDateLocator()
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(locator)
            )

    plt.tight_layout()
    plt.show()

def plot_loss_curves(loss_histories, look_back_windows, model_names, pred_len):
    """Plot available training and validation loss histories."""
    for look_back in look_back_windows:
        available_histories = {
            model_name: loss_histories[look_back][model_name]
            for model_name in model_names
            if loss_histories[look_back][model_name] is not None
        }

        # Loaded checkpoints from old runs do not contain loss history
        if not available_histories:
            print(
                f"Skipping loss plot for {look_back}-day look-back: "
                f"training history is unavailable."
            )
            continue

        plt.figure(figsize=(10, 5))

        for model_name, history in available_histories.items():
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
            f"-> {pred_len}-Day Forecast"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Normalized MSE Loss")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_test_predictions(test_results, look_back_windows, model_names, pred_len):
    """Plot the final horizon prediction for each look-back configuration."""
    rows = 2
    cols = 2

    figure, axes = plt.subplots(
        rows,
        cols,
        figsize=(18, 12),
    )

    figure.suptitle(
        f"VIC Test Predictions: Day-{pred_len} Forecast",
        fontsize=18,
        fontweight="bold",
    )

    for index, look_back in enumerate(look_back_windows):
        axis = axes[index // cols, index % cols]

        actual_values = test_results[
            look_back
        ][model_names[0]]["actuals"][:, -1]

        axis.plot(
            actual_values,
            linewidth=2,
            label="Actual",
            alpha=0.8,
        )

        for model_name in model_names:
            result = test_results[look_back][model_name]

            axis.plot(
                result["predictions"][:, -1],
                linewidth=1.5,
                linestyle="--",
                label=(
                    f"{model_name} "
                    f"(RMSE: {result['rmse']:,.0f})"
                ),
                alpha=0.8,
            )

        axis.set_title(
            f"{look_back}-Day Input -> "
            f"{pred_len}-Day Forecast"
        )
        axis.set_xlabel("Test Window")
        axis.set_ylabel("VIC Closing Price (VND)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_future_forecast(
    dataframe,
    forecast_dates,
    forecast_prices,
    model_name,
    look_back,
    recent_days=120,
):
    """Plot recent VIC prices together with future predictions."""
    recent_history = dataframe.tail(recent_days)

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
        label=f"Forecast: {model_name} ({look_back}-day look-back)",
    )

    plt.axvline(
        dataframe["time"].iloc[-1],
        linestyle=":",
        alpha=0.7,
        label="Forecast Start",
    )

    plt.title(
        "VIC Closing Price: Historical Data and Next 5-Day Forecast"
    )
    plt.xlabel("Date")
    plt.ylabel("VIC Closing Price (VND)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()