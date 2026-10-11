# VIC Stock Price Forecasting with LTSF-Linear Models

A time-series forecasting project that predicts the next 5 VIC stock closing prices using three lightweight forecasting models: **Linear**, **NLinear**, and **DLinear**.

The project evaluates multiple historical look-back windows and automatically selects the best-performing experiment based on test RMSE.

## Overview

This project applies Long-Term Series Forecasting techniques to historical VIC stock prices.

The forecasting task is:

* Input: historical VIC closing prices
* Look-back windows: `5`, `30`, `120`, and `480` trading days
* Forecast horizon: `5` future business days
* Models:

  * Linear
  * NLinear
  * DLinear
* Main evaluation metrics:

  * RMSE
  * MAE
  * R²

The target closing price is first log-transformed and standardized before being passed to the models. Predictions are converted back to the original VND price scale for evaluation and forecasting.

## Models

### Linear

The Linear model directly maps a historical sequence of length `seq_len` to the next `pred_len` values using a single linear layer.

```text
Historical Sequence
        ↓
    Linear Layer
        ↓
Future Predictions
```

### NLinear

NLinear normalizes each input sequence using its most recent observed value.

The model learns relative movements instead of absolute price levels.

```text
Input Sequence
      ↓
Subtract Last Value
      ↓
Linear Layer
      ↓
Add Last Value Back
      ↓
Prediction
```

### DLinear

DLinear decomposes the input sequence into:

* Trend component
* Seasonal component

Each component is processed by a separate linear layer before the outputs are combined.

```text
Input Sequence
      ↓
Moving Average Decomposition
      ↓
┌─────────────┬─────────────┐
│    Trend    │  Seasonal   │
│ Linear Layer│ Linear Layer│
└─────────────┴─────────────┘
      ↓
Combined Prediction
```

## Sliding-Window Dataset

Each training sample contains:

```text
X = historical prices
y = future prices
```

For example, with:

```text
look_back = 30
prediction_length = 5
```

the model receives:

```text
Past 30 trading days
        ↓
Predict
        ↓
Next 5 trading days
```

The project experiments with:

```text
5   → 5
30  → 5
120 → 5
480 → 5
```

## Evaluation

Each model is evaluated on the original VIC price scale using:

### Mean Absolute Error

```text
MAE = mean(|actual - predicted|)
```

### Root Mean Squared Error

```text
RMSE = sqrt(mean((actual - predicted)²))
```

### R² Score

R² measures how much of the variation in the target values is explained by the model.

The experiment with the lowest RMSE is selected as the best configuration.


## Possible Future Improvements

Potential extensions include:

* Add LSTM, GRU, Transformer models
* Add additional stock features such as volume, open, high, and low prices
* Add technical indicators
* Save complete training histories
* Add early stopping
* Add experiment tracking
* Add configuration files
* Create an inference-only application
* Build an interactive Streamlit dashboard
* Compare recursive and direct multi-step forecasting strategies
* Support forecasting for other stock symbols

## Disclaimer

This project is intended for educational and research purposes only.
