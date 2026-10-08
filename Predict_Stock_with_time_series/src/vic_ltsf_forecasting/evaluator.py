import numpy as np
import pandas as pd
import torch

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

@torch.no_grad()
def evaluate_predictions(model, dataloader, scaler, device):
    """Evaluate predictions on the original VIC price scale."""
    model.eval()

    all_predictions = []
    all_targets = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        predictions = model(batch_x)

        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(batch_y.numpy())

    predictions_scaled = np.concatenate(all_predictions, axis=0)
    targets_scaled = np.concatenate(all_targets, axis=0)

    predictions_log = scaler.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).reshape(predictions_scaled.shape)

    targets_log = scaler.inverse_transform(
        targets_scaled.reshape(-1, 1)
    ).reshape(targets_scaled.shape)

    # The model predicts log prices, so exponentiate to recover VND prices
    predictions_price = np.exp(predictions_log)
    targets_price = np.exp(targets_log)

    predictions_flat = predictions_price.flatten()
    targets_flat = targets_price.flatten()

    mse = mean_squared_error(targets_flat, predictions_flat)
    mae = mean_absolute_error(targets_flat, predictions_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_flat, predictions_flat)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": predictions_price,
        "actuals": targets_price,
    }

def create_performance_summary(test_results, look_back_windows, model_names):
    """Create a table containing metrics for all experiments."""
    rows = []

    for look_back in look_back_windows:
        for model_name in model_names:
            result = test_results[look_back][model_name]

            rows.append(
                {
                    "Look-back Days": look_back,
                    "Model": model_name,
                    "RMSE (VND)": result["rmse"],
                    "MAE (VND)": result["mae"],
                    "R²": result["r2"],
                }
            )

    return pd.DataFrame(rows).sort_values(
        by="RMSE (VND)",
        ascending=True,
    ).reset_index(drop=True)

def get_best_experiment(summary_df):
    """Return the experiment with the lowest RMSE."""
    best_index = summary_df["RMSE (VND)"].idxmin()
    return summary_df.loc[best_index]