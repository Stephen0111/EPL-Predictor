import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
TICKER = 'GOOG'
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODEL_FILE = os.path.join(MODEL_DIR, f'{TICKER}_predictor_model.joblib')
SCALER_FILE = os.path.join(MODEL_DIR, f'{TICKER}_predictor_scaler.joblib')
WINDOW_SIZE = 14  # Window for moving average and volatility calculation

def fetch_data(ticker):
    """Fetches historical daily data using yfinance."""
    print(f"Fetching data for {ticker}...")
    try:
        data = yf.download(ticker, period="2y", interval="1d")
        if data.empty:
            raise ValueError("No data returned from yfinance.")
        return data.dropna()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def feature_engineering(df):
    """Adds moving averages (features) and prepares the prediction target (label)."""
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
    return df

def train_and_save_model(df, ticker):
    """Trains a Linear Regression model and saves the model and scaler."""
    if df.empty:
        print("DataFrame is empty. Cannot train model.")
        return

    # Define features (X) and target (y)
    features = ['Close', 'MA_7', 'MA_14', 'Volume', 'Volatility']
    X = df[features].values
    y = df['Target'].values

    # --- Preprocessing: Scaling ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Training ---
    # Using the last 80% of data for training, reserving the rest for testing/validation
    split_index = int(0.8 * len(X_scaled))
    X_train = X_scaled[:split_index]
    y_train = y[:split_index]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- Saving Artifacts ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"\nSuccessfully trained Linear Regression model for {ticker}.")
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Scaler saved to: {SCALER_FILE}")

def run_training():
    """Main function to orchestrate data fetching, processing, and training."""
    # 1. Fetch and process data
    historical_data = fetch_data(TICKER)
    if historical_data.empty:
        print("Training halted due to data fetching error.")
        return

    processed_df = feature_engineering(historical_data.copy())
    
    # 2. Train and save
    train_and_save_model(processed_df, TICKER)
    
    # 3. Save processed data for initial database seeding (if needed)
    # Note: In a real environment, you would use this data to populate Firestore.
    # For this demonstration, we'll assume the client needs the processed data structure.
    processed_df.to_csv(os.path.join(MODEL_DIR, f'{TICKER}_processed_data.csv'))
    print(f"Processed data snapshot saved to {TICKER}_processed_data.csv")

if __name__ == '__main__':
    run_training()
