import pandas as pd

from vic_ltsf_forecasting.config import RESULT_DIR
from vic_ltsf_forecasting.visualization import (
    plot_rmse_comparison,
    plot_loss,
    plot_test_predictions
)

RUN_NAME = "run_01"


def main():
    run_dir = RESULT_DIR / RUN_NAME
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(run_dir / "summary.csv")
    plot_rmse_comparison(summary, figures_dir / "rmse_comparison.png")

    best = summary.loc[summary["rmse"].idxmin()]
    model_name = best["model"]
    look_back = int(best["look_back"])

    experiment_dir = run_dir / "experiments" / f"{model_name}_{look_back}d"

    history = pd.read_csv(experiment_dir / "history.csv")
    predictions = pd.read_csv(experiment_dir / "test_predictions.csv")

    plot_loss(history, model_name, look_back, figures_dir / "best_model_loss.png")
    plot_test_predictions(
        predictions,
        model_name,
        look_back,
        figures_dir / "best_model_predictions.png"
    )

    print(f"Figures saved at: {figures_dir}")


if __name__ == "__main__":
    main()