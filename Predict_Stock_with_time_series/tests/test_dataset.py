import numpy as np
from sklearn.preprocessing import StandardScaler

from vic_ltsf_forecasting.dataset import UnivariateTimeSeriesDataset

def test_dataset_output_shape():
    series = np.arange(
        100,
        dtype=np.float32,
    )

    dates = np.arange(100)

    scaler = StandardScaler()
    scaler.fit(
        series[:70].reshape(-1, 1)
    )

    dataset = UnivariateTimeSeriesDataset(
        series=series,
        dates=dates,
        seq_len=10,
        pred_len=5,
        start_window_index=0,
        end_window_index=50,
        scaler=scaler,
    )

    x, y = dataset[0]

    assert x.shape == (10,)
    assert y.shape == (5,)
    assert len(dataset) == 50