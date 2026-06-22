"""
Stock Forecaster MCP Server
============================

Install:
    pip install mcp yfinance

Run:
    python stock_forecaster_server.py

Claude Desktop config (claude_desktop_config.json):
    {
      "mcpServers": {
        "stock-forecaster": {
          "command": "python",
          "args": ["/absolute/path/to/stock_forecaster_server.py"]
        }
      }
    }
"""
import os
import random
from datetime import date, timedelta
from typing import Any
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yfinance as yf

from mcp.server.fastmcp import FastMCP

# Server setup
mcp = FastMCP(
    name="StockForecaster",
    instructions=(
        "Provides short-term stock price forecasts using an LSTM model trained "
        "on OHLCV data and technical indicators (RSI, MACD, Bollinger Bands, ATR). "
        "Call get_forecast with a ticker symbol and the number of trading days to forecast."
    ),
)

# Load model and scalers at startup


class StockLSTM(nn.Module):
    def __init__(self, input_size=13, hidden_size=64, num_layers=2,
                 dropout=0.2, forecast_n=5):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, forecast_n)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(self.dropout(lstm_out[:, -1, :]))

# Load checkpoint and scalers
_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_CHECKPOINT = os.path.join(_BASE_DIR, "best_model.pt")
_SCALERS    = os.path.join(_BASE_DIR, "scalers.pkl")

_ckpt   = torch.load(_CHECKPOINT, map_location="cpu")
_config = _ckpt["config"]
_model  = StockLSTM(**{k: _config[k] for k in
            ["input_size","hidden_size","num_layers","dropout","forecast_n"]})
_model.load_state_dict(_ckpt["model_state"])
_model.eval()

with open(_SCALERS, "rb") as f:
    _scalers = pickle.load(f)

_FEATURE_COLS = _config["feature_cols"]
_FORECAST_N   = _config["forecast_n"]
_WINDOW_SIZE  = 60

def _run_real_inference(ticker: str, days: int) -> list[dict]:
    import ta
    # Fetch recent data — grab extra rows for indicator warmup
    df = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    # Compute indicators
    close, high, low = df["close"], df["high"], df["low"]
    df["rsi"]         = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd              = ta.trend.MACD(close=close)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()
    bb                = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df["bb_high"]     = bb.bollinger_hband()
    df["bb_mid"]      = bb.bollinger_mavg()
    df["bb_low"]      = bb.bollinger_lband()
    df["atr"]         = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    df.dropna(inplace=True)

    # Use the closest ticker scaler (fallback to AAPL if ticker not in training set)
    scaler_key = ticker if ticker in _scalers else list(_scalers.keys())[0]
    scaler     = _scalers[scaler_key]

    arr    = df[_FEATURE_COLS].values[-_WINDOW_SIZE:]
    scaled = scaler.transform(arr)
    x      = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # (1, 60, 13)

    with torch.no_grad():
        pred_scaled = _model(x).squeeze(0).numpy()  # (forecast_n,)

    # Inverse transform — only the close column
    close_idx  = _FEATURE_COLS.index("close")
    dummy      = np.zeros((len(pred_scaled), len(_FEATURE_COLS)))
    dummy[:, close_idx] = pred_scaled
    pred_prices = scaler.inverse_transform(dummy)[:, close_idx]

    # Build output
    results      = []
    current_date = date.today()
    for i, price in enumerate(pred_prices[:days]):
        current_date += timedelta(days=1)
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        band = round(float(price) * 0.02 * (1 + i * 0.1), 2)
        results.append({
            "date":            current_date.isoformat(),
            "predicted_close": round(float(price), 2),
            "lower_bound":     round(float(price) - band, 2),
            "upper_bound":     round(float(price) + band, 2),
        })
    return results


# MCP tool

VALID_TICKERS = {"AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "META", "NVDA", "AMD"}

@mcp.tool()
def get_forecast(ticker: str, days: int = 5) -> dict[str, Any]:
    """
    Forecast the closing price of a stock for the next N trading days.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'MSFT', 'TSLA').
        days:   Number of trading days to forecast. Must be between 1 and 30.

    Returns:
        A dictionary with the ticker, model info, and a list of daily forecasts.
        Each forecast contains: date, predicted_close, lower_bound, upper_bound.
    """
    # Input validation
    ticker = ticker.upper().strip()
    if not ticker:
        raise ValueError("Ticker symbol cannot be empty.")
    if days < 1 or days > 30:
        raise ValueError(f"days must be between 1 and 30, got {days}.")

    # Inference 
    forecast_points = _run_real_inference(ticker, days)

    return {
        "ticker": ticker,
        "forecast_days": days,
        "model": "LSTM-v1",          
        "generated_at": date.today().isoformat(),
        "forecast": forecast_points,
        "disclaimer": (
            "This is a machine learning forecast for educational purposes only. "
            "It is not financial advice."
        ),
    }

# Second tool for model metadata (useful for your notebook demo)


@mcp.tool()
def get_model_info() -> dict[str, Any]:
    """
    Returns metadata about the forecasting model currently loaded.
    Useful for verifying which model version is active.
    """
    return {
        "model_name": "LSTM-v1",
        "status": "trained LSTM checkpoint loaded",
        "input_features": [
            "open", "high", "low", "close", "volume",
            "RSI", "MACD", "BollingerBands", "ATR",
        ],
        "output": "predicted closing price per trading day",
        "trained_tickers": list(VALID_TICKERS),
    }

# Entry point

if __name__ == "__main__":
    # stdio transport is required for Claude Desktop
    mcp.run(transport="stdio")
