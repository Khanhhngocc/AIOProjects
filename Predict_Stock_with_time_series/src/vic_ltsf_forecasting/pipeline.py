import joblib

from vic_ltsf_forecasting.config import (
    TARGET_COLUMN,
    LOOK_BACK_WINDOWS,
    PREDICTION_LENGTH,
    TRAIN_RATIO,
    VALID_RATIO,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    MOVING_AVERAGE_WINDOW,
    MODEL_NAMES,
    MODEL_DIR,
    RESULT_DIR,
    DEVICE,
    SEED,
)

from vic_ltsf_forecasting.utils import set_seed
from vic_ltsf_forecasting.data_loader import (
    load_dataframe,
    prepare_dataframe,
    prepare_series_and_scaler,
)
from vic_ltsf_forecasting.dataset import build_datasets_and_loaders
from vic_ltsf_forecasting.models import create_model
from vic_ltsf_forecasting.trainer import (
    train_model,
    load_model_from_checkpoint,
)
from vic_ltsf_forecasting.evaluator import (
    evaluate_predictions,
    create_performance_summary,
    get_best_experiment,
)
from vic_ltsf_forecasting.forecaster import (
    forecast_next_days,
    create_forecast_dataframe,
)
from vic_ltsf_forecasting.visualization import (
    plot_eda,
    plot_loss_curves,
    plot_test_predictions,
    plot_future_forecast,
)


def run_pipeline(
    num_epochs=NUM_EPOCHS,
    show_plots=True,
    force_train=False,
):
    """Run the complete VIC forecasting pipeline."""
    set_seed(SEED)

    # Automatically recreate output directories if they are missing
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")

    # 1. Load and preprocess data
    df = load_dataframe()
    df = prepare_dataframe(df)

    print(f"\nDataset shape: {df.shape}")
    print(
        f"Date range: "
        f"{df['time'].min().date()} -> "
        f"{df['time'].max().date()}"
    )

    if show_plots:
        plot_eda(df)

    # 2. Create chronological split and fit the scaler on training data
    series, train_end, valid_end, scaler = prepare_series_and_scaler(
        dataframe=df,
        target_column=TARGET_COLUMN,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VALID_RATIO,
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "vic_standard_scaler.pkl",
    )

    print("\nTime-based split:")
    print(f"Train: 0 -> {train_end - 1}")
    print(f"Valid: {train_end} -> {valid_end - 1}")
    print(f"Test : {valid_end} -> {len(series) - 1}")

    # 3. Create Dataset and DataLoader objects
    datasets_by_window, loaders_by_window = build_datasets_and_loaders(
        series=series,
        dates=df["time"].to_numpy(),
        look_back_windows=LOOK_BACK_WINDOWS,
        pred_len=PREDICTION_LENGTH,
        train_end=train_end,
        valid_end=valid_end,
        scaler=scaler,
        batch_size=BATCH_SIZE,
    )

    # 4. Load existing checkpoints or train missing models
    trained_models = {}
    loss_histories = {}
    test_results = {}

    for look_back in LOOK_BACK_WINDOWS:
        trained_models[look_back] = {}
        loss_histories[look_back] = {}
        test_results[look_back] = {}

        train_loader = loaders_by_window[look_back]["train"]
        valid_loader = loaders_by_window[look_back]["valid"]
        test_loader = loaders_by_window[look_back]["test"]

        print(f"\n{'=' * 70}")
        print(
            f"Look-back: {look_back} days -> "
            f"Forecast: {PREDICTION_LENGTH} days"
        )
        print(f"{'=' * 70}")

        for model_name in MODEL_NAMES:
            set_seed(SEED)

            model = create_model(
                model_name=model_name,
                seq_len=look_back,
                pred_len=PREDICTION_LENGTH,
                moving_avg=MOVING_AVERAGE_WINDOW,
            )

            checkpoint_path = MODEL_DIR / (
                f"vic_{model_name.lower()}_{look_back}d_best.pt"
            )

            if checkpoint_path.exists() and not force_train:
                print(
                    f"\nLoading checkpoint -> "
                    f"{checkpoint_path.name}"
                )

                trained_model, checkpoint = load_model_from_checkpoint(
                    model=model,
                    checkpoint_path=checkpoint_path,
                    device=DEVICE,
                )

                history = None

                print(
                    f"Loaded {model_name} | "
                    f"Best epoch: {checkpoint['epoch']} | "
                    f"Best valid MSE: "
                    f"{checkpoint['best_valid_loss']:.6f}"
                )

            else:
                if force_train and checkpoint_path.exists():
                    print(
                        f"\nForce training -> "
                        f"{model_name}, {look_back}-day look-back"
                    )
                else:
                    print(
                        f"\nCheckpoint not found. Training -> "
                        f"{model_name}, {look_back}-day look-back"
                    )

                trained_model, history, checkpoint_path = train_model(
                    model=model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    model_name=model_name,
                    look_back=look_back,
                    pred_len=PREDICTION_LENGTH,
                    device=DEVICE,
                    num_epochs=num_epochs,
                    learning_rate=LEARNING_RATE,
                )

            # Evaluate loaded or newly trained model on the test set
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
                f"Test -> {model_name} | "
                f"RMSE: {result['rmse']:,.0f} VND | "
                f"MAE: {result['mae']:,.0f} VND | "
                f"R²: {result['r2']:.4f}"
            )

    # 5. Compare all experiments
    summary_df = create_performance_summary(
        test_results=test_results,
        look_back_windows=LOOK_BACK_WINDOWS,
        model_names=MODEL_NAMES,
    )

    summary_path = (
        RESULT_DIR / "vic_ltsf_performance_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print("\nTest Performance Summary:")
    print(summary_df.to_string(index=False))

    # 6. Select the experiment with the lowest RMSE
    best_row = get_best_experiment(summary_df)

    best_look_back = int(best_row["Look-back Days"])
    best_model_name = best_row["Model"]
    best_model = trained_models[
        best_look_back
    ][best_model_name]

    print("\nBest Experiment:")
    print(f"Model     : {best_model_name}")
    print(f"Look-back : {best_look_back} days")
    print(
        f"RMSE      : "
        f"{best_row['RMSE (VND)']:,.0f} VND"
    )

    # 7. Forecast the next business days
    forecast_dates, forecast_prices = forecast_next_days(
        dataframe=df,
        model=best_model,
        scaler=scaler,
        look_back=best_look_back,
        pred_len=PREDICTION_LENGTH,
        device=DEVICE,
    )

    forecast_df = create_forecast_dataframe(
        forecast_dates,
        forecast_prices,
    )

    forecast_path = (
        RESULT_DIR / "vic_next_5_day_forecast.csv"
    )

    forecast_df.to_csv(
        forecast_path,
        index=False,
    )

    print("\nNext 5-Day Forecast:")
    print(forecast_df.to_string(index=False))

    # 8. Visualize results
    if show_plots:
        plot_loss_curves(
            loss_histories=loss_histories,
            look_back_windows=LOOK_BACK_WINDOWS,
            model_names=MODEL_NAMES,
            pred_len=PREDICTION_LENGTH,
        )

        plot_test_predictions(
            test_results=test_results,
            look_back_windows=LOOK_BACK_WINDOWS,
            model_names=MODEL_NAMES,
            pred_len=PREDICTION_LENGTH,
        )

        plot_future_forecast(
            dataframe=df,
            forecast_dates=forecast_dates,
            forecast_prices=forecast_prices,
            model_name=best_model_name,
            look_back=best_look_back,
        )

    return {
        "dataframe": df,
        "datasets": datasets_by_window,
        "models": trained_models,
        "histories": loss_histories,
        "test_results": test_results,
        "summary": summary_df,
        "forecast": forecast_df,
    }