


import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import traceback

# --- Configuration ---
# Define ALL tickers to be processed
TOP_TICKERS = ['GOOG', 'MSFT', 'AAPL', 'TSLA', 'AMZN']
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
WINDOW_SIZE = 14  # Window for moving average and volatility calculation

def fetch_data(ticker):
    """Fetches historical daily data using yfinance."""
    print(f"Fetching data for {ticker}...")
    try:
        # Fetch 2 years of data to ensure enough history for feature creation
        # progress=False to suppress yfinance progress bars
        data = yf.download(ticker, period="2y", interval="1d", progress=False)
        
        if data.empty:
            raise ValueError("No data returned from yfinance.")
        
        # Fix MultiIndex columns issue - flatten if necessary
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        print(f"Successfully fetched {len(data)} rows for {ticker}")
        return data.dropna()
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def feature_engineering(df):
    """Adds moving averages (features) and prepares the prediction target (label)."""
    
    print(f"  Running feature engineering on {len(df)} rows...")
    
    # 1. Target Variable (The price we are trying to predict)
    # The 'Close' price shifted by -1 is the next day's closing price
    df['Target'] = df['Close'].shift(-1)

    # 2. Features: Moving Averages
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_14'] = df['Close'].rolling(window=14).mean()
    
    # 3. Features: Volatility (Rolling Standard Deviation of returns)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=WINDOW_SIZE).std()

    # Drop the last row (where Target is NaN) and initial rows (where MAs are NaN)
    df = df.dropna()
    print(f"  After feature engineering: {len(df)} rows")
    return df

def train_and_save_model(df, ticker):
    """Trains a Linear Regression model and saves the model and scaler."""
    if df.empty:
        print(f"DataFrame for {ticker} is empty. Cannot train model.")
        return

    print(f"  Training model for {ticker}...")
    
    # Define features (X) and target (y)
    features = ['Close', 'MA_7', 'MA_14', 'Volume', 'Volatility']
    X = df[features].values
    y = df['Target'].values

    # --- Preprocessing: Scaling ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Training ---
    # Using the last 80% of data for training
    split_index = int(0.8 * len(X_scaled))
    X_train = X_scaled[:split_index]
    y_train = y[:split_index]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- Saving Artifacts ---
    model_file = os.path.join(MODEL_DIR, f'{ticker}_predictor_model.joblib')
    scaler_file = os.path.join(MODEL_DIR, f'{ticker}_predictor_scaler.joblib')
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    print(f"  [SUCCESS] Trained and saved Linear Regression model for {ticker}.")

def run_training_batch():
    """Main function to orchestrate data fetching, processing, and training for all tickers."""
    
    print("--- Starting Batch Training for Financial Models ---")
    print(f"Tickers to process: {TOP_TICKERS}")
    print(f"Total tickers: {len(TOP_TICKERS)}\n")
    
    success_count = 0
    failed_tickers = []
    
    for ticker in TOP_TICKERS:
        try:
            print(f"\n{'='*60}")
            print(f"### Processing Ticker: {ticker} ###")
            print(f"{'='*60}")
            
            # 1. Fetch and process data
            historical_data = fetch_data(ticker)
            if historical_data.empty:
                print(f"❌ Skipping training for {ticker} due to data error.")
                failed_tickers.append(ticker)
                continue

            processed_df = feature_engineering(historical_data.copy())
            
            if processed_df.empty:
                print(f"❌ Skipping training for {ticker} - no data after feature engineering.")
                failed_tickers.append(ticker)
                continue
            
            # 2. Train and save
            train_and_save_model(processed_df, ticker)
            
            # 3. Save processed data for debugging/snapshot
            csv_path = os.path.join(MODEL_DIR, f'{ticker}_processed_data.csv')
            processed_df.to_csv(csv_path)
            print(f"  Saved processed data to: {csv_path}")
            
            success_count += 1
            print(f"✓ Successfully completed {ticker}")
            
        except Exception as e:
            print(f"\n❌ ERROR processing {ticker}: {str(e)}")
            traceback.print_exc()
            failed_tickers.append(ticker)
            print(f"Continuing to next ticker...\n")
            continue
    
    print(f"\n{'='*60}")
    print("--- Batch Training Complete ---")
    print(f"{'='*60}")
    print(f"✓ Successfully processed: {success_count}/{len(TOP_TICKERS)} tickers")
    if failed_tickers:
        print(f"❌ Failed tickers: {', '.join(failed_tickers)}")
    else:
        print("✓ All tickers processed successfully!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    run_training_batch()