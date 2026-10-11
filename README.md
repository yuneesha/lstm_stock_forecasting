# Stock Price Forecasting with LSTM Networks

Deep learning pipeline that forecasts daily closing prices for three coffee-chain stocks: **Starbucks (SBUX)**, **Luckin Coffee (LKNCY)** and **Dutch Bros (BROS)**. It uses a tuned stacked LSTM model built in TensorFlow/Keras.

> Academic project for **DA4131 – Advanced ML Applications for Business**, University of Moratuwa.
### Group Members

#### Gihini Waidyalankara | Yuneesha Ediriweera | Sanali Wimalarathne

![Forecast](images/forecast_all.png)

---

## Problem

Can a sequence model trained on five years of price history produce reliable short-term forecasts to support **buy / sell / hold** decisions?

- **Data period:** 31 March 2021 – 31 March 2026 (daily OHLCV from Yahoo Finance)
- **Forecast window:** 1 – 14 May 2026 (10 trading days)
- **Note:** BROS has about 4.5 years of data because it listed on the NYSE on 15 September 2021.

## Approach

```
Data collection → EDA → Preprocessing → LSTM design → Tuning & evaluation → Forecasting
```

1. **Data collection:** downloaded OHLCV data with `yfinance`.
2. **EDA:** price history, weekly volume, 30/90-day moving averages, return distributions (volatility, skew, kurtosis) and correlation heatmaps, with price moves linked to real events (e.g., the SBUX CEO change in Aug 2024 and the Chinese ADR delisting scare in Mar 2022).
3. **Preprocessing:** closing prices scaled with `StandardScaler`, 90-day sliding windows, chronological 80/20 train/test split (no shuffling; test period roughly Apr 2025 – Mar 2026). Outliers were kept because they are genuine market events.
4. **Model:** stacked LSTM → Dense(64, ReLU) → Dense(1), with dropout, Adam optimiser, MAE loss, EarlyStopping and ReduceLROnPlateau.
5. **Hyperparameter tuning:** compared 4 configurations on validation MAE:

   | Trial | Config | LSTM units | Dropout | Lookback | Val MAE |
   |---|---|---|---|---|---|
   | 1 | Baseline | (64, 64) | 0.5 | 60 | 0.1117 |
   | 2 | Initial model | (128, 64, 32) | 0.3 | 90 | 0.1142 |
   | 3 | Lower dropout | (128, 64, 32) | 0.2 | 90 | 0.1077 |
   | **4** | **Larger units (selected)** | **(256, 128, 64)** | **0.3** | **90** | **0.1071** |

6. **Forecasting:** recursive (auto-regressive) 10-step forecast, where each prediction is fed back into the input window.

## Results

Test-set performance of the tuned model (last 20% of data, about Apr 2025 – Mar 2026):

| Stock | MAE (USD) | RMSE (USD) | R² |
|---|---|---|---|
| SBUX | 1.37 | 1.91 | **0.888** |
| LKNCY | 0.81 | 1.10 | **0.846** |
| BROS | 2.41 | 3.10 | **0.805** |

- SBUX was the easiest to predict because it is the least volatile of the three (about 2% daily std dev).
- Tuning improved R² for all three stocks. BROS gained the most (0.719 → 0.805).
- RMSE > MAE for all three, so the model struggles on sudden, news-driven price jumps.

![Actual vs predicted](images/actual_vs_predicted.png)

**May 2026 forecast** (full results in [`outputs/Output.csv`](outputs/Output.csv)):

| Stock | 1 May | 14 May |
|---|---|---|
| SBUX | $106.46 | $105.44 |
| LKNCY | $34.97 | $33.62 |
| BROS | $56.20 | $56.09 |

![Training curves](images/training_curves.png)

## Repository Structure

```
├── DA4131_LSTM_Stock_Prediction.ipynb   # Full pipeline notebook
├── requirements.txt
├── data/                                # Raw OHLCV data (SBUX, LKNCY, BROS)
├── outputs/Output.csv                   # 10-day price forecasts
└── images/                              # Key charts
```

## How to Run

```bash
git clone https://github.com/<your-username>/lstm-stock-forecasting.git
cd lstm-stock-forecasting
pip install -r requirements.txt
jupyter notebook DA4131_LSTM_Stock_Prediction.ipynb
```

You can also open the notebook in **Google Colab**, where most dependencies are already installed.

## Tech Stack

Python · TensorFlow / Keras · scikit-learn · Pandas · NumPy · yfinance · Plotly · Matplotlib · Seaborn

## Limitations

- The model uses **closing price only**. It has no volume, technical indicators, news or sentiment inputs.
- Recursive forecasting compounds errors, so forecasts flatten out and become less reliable further ahead.
- The model can't anticipate earnings surprises, leadership changes or regulatory events.

## Disclaimer

This is an academic project and **not financial advice**.
