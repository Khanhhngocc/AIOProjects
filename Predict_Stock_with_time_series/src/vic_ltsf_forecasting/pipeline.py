import pandas as pd

from vic_ltsf_forecasting.config import (
    RESULT_DIR, MODEL_DIR, MODEL_NAMES, LOOK_BACK_WINDOWS,
    PREDICTION_LENGTH, NUM_EPOCHS, LEARNING_RATE,
    BATCH_SIZE, SEED, DEVICE
)
from vic_ltsf_forecasting.data_loader import load_dataframe, prepare_series
from vic_ltsf_forecasting.dataset import create_loaders
from vic_ltsf_forecasting.models import create_model
from vic_ltsf_forecasting.trainer import train_model
from vic_ltsf_forecasting.evaluator import evaluate_model
from vic_ltsf_forecasting.utils import (
    set_seed, create_run_dir, save_json, save_dataframe
)


def run_training_pipeline():
    # 1. Create run
    set_seed(SEED)

    run_dir = create_run_dir(RESULT_DIR)
    model_run_dir = MODEL_DIR / run_dir.name
    model_run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "models": MODEL_NAMES,
        "look_back_windows": LOOK_BACK_WINDOWS,
        "prediction_length": PREDICTION_LENGTH,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "seed": SEED
    }

    save_json(config, run_dir / "config.json")

    # 2. Load data
    dataframe = load_dataframe()
    series, scaler, train_end, valid_end = prepare_series(dataframe)
    loaders = create_loaders(series, scaler, train_end, valid_end)

    print(f"Device: {DEVICE}")
    print(f"Run: {run_dir.name}")
    print(f"Dataset rows: {len(dataframe)}")

    # 3. Train experiments
    summary_rows = []

    for look_back in LOOK_BACK_WINDOWS:
        for model_name in MODEL_NAMES:
            print(f"\nTraining {model_name} | Look-back: {look_back} days")

            set_seed(SEED)

            model = create_model(model_name, look_back, PREDICTION_LENGTH)

            checkpoint_path = (
                model_run_dir /
                f"vic_{model_name.lower()}_{look_back}d_best.pt"
            )

            model, history, best_valid_loss, best_epoch = train_model(
                model,
                loaders[look_back]["train"],
                loaders[look_back]["valid"],
                checkpoint_path,
                DEVICE,
                NUM_EPOCHS,
                LEARNING_RATE
            )

            result = evaluate_model(
                model,
                loaders[look_back]["test"],
                scaler,
                DEVICE
            )

            experiment_dir = (
                run_dir /
                "experiments" /
                f"{model_name}_{look_back}d"
            )

            experiment_dir.mkdir(parents=True, exist_ok=True)

            history_df = pd.DataFrame({
                "epoch": range(1, NUM_EPOCHS + 1),
                "train_loss": history["train_loss"],
                "valid_loss": history["valid_loss"]
            })

            save_dataframe(
                history_df,
                experiment_dir / "history.csv"
            )

            metrics = {
                "model": model_name,
                "look_back": look_back,
                "best_epoch": best_epoch,
                "best_valid_loss": best_valid_loss,
                "rmse": result["rmse"],
                "mae": result["mae"],
                "r2": result["r2"]
            }

            save_json(
                metrics,
                experiment_dir / "metrics.json"
            )

            prediction_rows = []

            for window in range(len(result["predictions"])):
                for horizon in range(PREDICTION_LENGTH):
                    prediction_rows.append({
                        "window": window,
                        "horizon_day": horizon + 1,
                        "actual": result["actuals"][window, horizon],
                        "predicted": result["predictions"][window, horizon]
                    })

            save_dataframe(
                pd.DataFrame(prediction_rows),
                experiment_dir / "test_predictions.csv"
            )

            summary_rows.append(metrics)

            print(
                f"Test -> RMSE: {result['rmse']:,.0f} VND | "
                f"MAE: {result['mae']:,.0f} VND | "
                f"R²: {result['r2']:.4f}"
            )

    # 4. Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("rmse").reset_index(drop=True)

    save_dataframe(
        summary_df,
        run_dir / "summary.csv"
    )

    best = summary_df.iloc[0]

    print("\nBest experiment:")
    print(f"Model: {best['model']}")
    print(f"Look-back: {int(best['look_back'])} days")
    print(f"RMSE: {best['rmse']:,.0f} VND")
    print(f"MAE: {best['mae']:,.0f} VND")
    print(f"R²: {best['r2']:.4f}")

    return run_dir