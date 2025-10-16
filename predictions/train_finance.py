
import yfinance as yf
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBRegressor
import ta
from datetime import datetime

# --- Configuration ---
TICKER = "GOOG"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_FILE = os.path.join(MODEL_DIR, f"{TICKER}_xgb_model.joblib")
SCALER_FILE = os.path.join(MODEL_DIR, f"{TICKER}_xgb_scaler.joblib")
WINDOW_SIZE = 14

def fetch_data(ticker):
    """Fetch 5 years of daily data."""
    data = yf.download(ticker, period="5y", interval="1d")
    if data.empty:
        raise ValueError("No data returned from yfinance.")
    return data.dropna()

def feature_engineering(df):
    """Add indicators and target."""
    df["Daily_Return"] = df["Close"].pct_change()
    df["Target"] = df["Daily_Return"].shift(-1)

    # Indicators
    df["MA_7"] = df["Close"].rolling(7).mean()
    df["MA_14"] = df["Close"].rolling(14).mean()
    df["Volatility"] = df["Daily_Return"].rolling(WINDOW_SIZE).std()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["EMA_9"] = ta.trend.EMAIndicator(df["Close"], window=9).ema_indicator()
    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    for lag in range(1, 4):
        df[f"Lag_{lag}"] = df["Close"].shift(lag)

    return df.dropna()

def train_model(df):
    """Train XGBoost model."""
    features = [
        "Close", "MA_7", "MA_14", "Volatility", "RSI", "MACD",
        "EMA_9", "BB_High", "BB_Low", "Lag_1", "Lag_2", "Lag_3", "Volume"
    ]
    X, y = df[features].values, df["Target"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, objective="reg:squarederror"
    )

    print("Performing time-series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring="neg_mean_absolute_error")
    print(f"Average MAE: {-np.mean(cv_scores):.6f}")

    model.fit(X_scaled, y)
    return model, scaler

def run_training():
    print(f"\n[{datetime.now()}] Starting training for {TICKER}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = fetch_data(TICKER)
    df = feature_engineering(df)
    model, scaler = train_model(df)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    df.to_csv(os.path.join(MODEL_DIR, f"{TICKER}_processed_v2.csv"))

    print(f"\n✅ Model retrained and saved to {MODEL_DIR}")
    print(f"Latest data date: {df.index[-1].date()}")

if __name__ == "__main__":
    run_training()
