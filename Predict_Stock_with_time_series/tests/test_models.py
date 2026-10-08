import torch

from vic_ltsf_forecasting.models import Linear, NLinear, DLinear

def test_linear_output_shape():
    model = Linear(
        seq_len=30,
        pred_len=5,
    )

    x = torch.randn(8, 30)
    output = model(x)

    assert output.shape == (8, 5)

def test_nlinear_output_shape():
    model = NLinear(
        seq_len=30,
        pred_len=5,
    )

    x = torch.randn(8, 30)
    output = model(x)

    assert output.shape == (8, 5)

def test_dlinear_output_shape():
    model = DLinear(
        seq_len=30,
        pred_len=5,
        moving_avg=5,
    )

    x = torch.randn(8, 30)
    output = model(x)

    assert output.shape == (8, 5)