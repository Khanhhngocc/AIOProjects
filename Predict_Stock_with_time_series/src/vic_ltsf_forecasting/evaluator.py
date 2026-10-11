import numpy as np
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


@torch.no_grad()
def evaluate_model(model, dataloader, scaler, device):
    model.eval()

    all_predictions = []
    all_actuals = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)

        predictions = model(batch_x)

        all_predictions.append(
            predictions.cpu().numpy()
        )

        all_actuals.append(
            batch_y.numpy()
        )

    predictions = np.concatenate(all_predictions)
    actuals = np.concatenate(all_actuals)

    predictions_log = scaler.inverse_transform(
        predictions.reshape(-1, 1)
    ).reshape(predictions.shape)

    actuals_log = scaler.inverse_transform(
        actuals.reshape(-1, 1)
    ).reshape(actuals.shape)

    predictions_price = np.exp(predictions_log)
    actuals_price = np.exp(actuals_log)

    prediction_flat = predictions_price.flatten()
    actual_flat = actuals_price.flatten()

    mse = mean_squared_error(
        actual_flat,
        prediction_flat
    )

    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(
            actual_flat,
            prediction_flat
        ),
        "r2": r2_score(
            actual_flat,
            prediction_flat
        ),
        "predictions": predictions_price,
        "actuals": actuals_price
    }