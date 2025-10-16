

import joblib
import os
import uvicorn
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
# from sqlalchemy.orm import Session  # REMOVED
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
from fastapi.responses import HTMLResponse, FileResponse
import yfinance as yf
from google.cloud.firestore import Client, Query
from firebase_admin import firestore
import ta

# Load environment variables
load_dotenv()
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
# --- Internal Imports (Now using Firestore structure) ---
from database import init_db, get_db, EPL_MATCHES_COLLECTION, EPL_TABLE_COLLECTION 
# Note: EPLMatch, EPLTable, SessionLocal removed

# --- Setup ---

app = FastAPI(title="EPL/Financial Predictor API")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
# Initialize database on startup 
init_db()

# Configure Templates for serving the frontend HTML
templates = Jinja2Templates(directory="templates")

# Configure CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/predictions", StaticFiles(directory="predictions"), name="predictions")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Schemas (Unchanged) ---

class TeamPredictionFeatures(BaseModel):
    """Input structure for prediction endpoint."""
    home_pts_last_5: int
    away_pts_last_5: int

class TeamPrediction(BaseModel):
    """Output structure for prediction endpoint."""
    home_team: str
    away_team: str
    probabilities: Dict[str, float]
    prediction: str

FIREBASE_CREDENTIALS_VAR = "FIREBASE_CREDENTIALS_JSON" 
# --- Model Loading & Database Configuration ---

# EPL Model Configuration (Unchanged)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'predictor_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'predictor_scaler.joblib')

PREDICTOR_MODEL = None
SCALER = None

# Financial Analytics Configuration (Unchanged)
SQLITE_FINANCE_DB_PATH = 'finance_data.db' 
TOP_TICKERS = ['GOOG', 'MSFT', 'AAPL', 'TSLA', 'AMZN']
FINANCE_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models') 

# In-memory storage for ALL loaded financial models and scalers
FIN_MODELS: Dict[str, Any] = {}
FIN_SCALERS: Dict[str, Any] = {}


# --- Auto-Update Functions (MODIFIED FOR FIRESTORE) ---

def update_league_standings(api_token: str, db: Client): # ADDED db: Client
    """
    Fetches the current EPL standings and updates the EPLTable collection in Firestore.
    """
    logger.info("[Startup] Updating current league standings...")
    
    if not api_token:
        logger.warning("Warning: FOOTBALL_DATA_API_TOKEN not set. Skipping standings update.")
        return None
    
    headers = {"X-Auth-Token": api_token}
    base_url = "https://api.football-data.org/v4"
    
    try:
        standings_url = f"{base_url}/competitions/PL/standings"
        standings_response = requests.get(standings_url, headers=headers, timeout=10)
        standings_response.raise_for_status()
        standings_data = standings_response.json()
        
        if 'standings' in standings_data:
            current_season_id = standings_data.get('season', {}).get('id')
            
            # --- FIRESTORE CHANGE: Batch Delete and Write ---
            batch = db.batch()
            table_ref = db.collection(EPL_TABLE_COLLECTION)
            
            # 1. Clear old standings (Firestore delete is done in chunks/transactions)
            docs = table_ref.stream()
            for doc in docs:
                batch.delete(doc.reference)
            
            team_count = 0
            for table in standings_data['standings']:
                if table['type'] == 'TOTAL': 
                    for team_info in table.get('table', []):
                        # 2. Prepare new document data
                        standing_doc = {
                            "season": current_season_id,
                            "position": team_info.get('position'),
                            "team": team_info['team']['name'],
                            "played": team_info.get('playedGames'),
                            "points": team_info.get('points'),
                            "goal_difference": team_info.get('goalDifference'),
                        }
                        # Use team name as document ID for easy reference (optional)
                        doc_ref = table_ref.document(team_info['team']['name'].replace(" ", "_"))
                        batch.set(doc_ref, standing_doc)
                        team_count += 1

            batch.commit()
            logger.info(f"[Startup] Successfully updated {team_count} league standings in Firestore.")
            return current_season_id

    except requests.exceptions.RequestException as e:
        logger.error(f"[Startup] Error fetching standings: {e}")
    except Exception as e:
        logger.error(f"[Startup] Error updating standings: {e}")
    
    return None


def update_current_season_matches(api_token: str, db: Client):
    """
    Fetches finished current season matches and updates the EPLMatch collection.
    """
    logger.info("[Startup] Updating current season matches...")
    
    if not api_token:
        logger.warning("Warning: FOOTBALL_DATA_API_TOKEN not set. Skipping matches update.")
        return
    
    headers = {"X-Auth-Token": api_token}
    base_url = "https://api.football-data.org/v4"
    
    try:
        matches_response = requests.get(
            f"{base_url}/competitions/PL/matches",
            headers=headers,
            timeout=10
        )
        matches_response.raise_for_status()
        matches_data = matches_response.json()
        
        matches_saved = 0
        batch = db.batch()
        matches_ref = db.collection(EPL_MATCHES_COLLECTION)
        
        for match in matches_data.get('matches', []):
            try:
                if match['status'] != 'FINISHED':
                    continue
                
                season = match['season']['id']
                home_team = match['homeTeam']['name']
                away_team = match['awayTeam']['name']
                home_goals = match['score']['fullTime']['home']
                away_goals = match['score']['fullTime']['away']
                match_date = match['utcDate']
                
                result = 'H' if home_goals > away_goals else ('A' if home_goals < away_goals else 'D')
                
                # --- FIXED: Correct Firestore query syntax ---
                existing_match_query = matches_ref\
                    .where("home_team", "==", home_team)\
                    .where("away_team", "==", away_team)\
                    .where("match_date", "==", match_date)\
                    .limit(1)\
                    .get()
                
                if existing_match_query:  # if list is not empty
                    continue
                
                # --- Get team points from EPLTable collection ---
                home_standing_doc = db.collection(EPL_TABLE_COLLECTION).document(home_team.replace(" ", "_")).get()
                away_standing_doc = db.collection(EPL_TABLE_COLLECTION).document(away_team.replace(" ", "_")).get()
                
                home_pts = home_standing_doc.get('points') if home_standing_doc.exists else 0
                away_pts = away_standing_doc.get('points') if away_standing_doc.exists else 0
                
                # Prepare document data
                match_doc = {
                    "season": season,
                    "match_date": match_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "full_time_home_goals": home_goals,
                    "full_time_away_goals": away_goals,
                    "result": result,
                    "home_pts_last_5": home_pts,
                    "away_pts_last_5": away_pts
                }
                
                # Use a compound key for the document ID
                doc_id = f"{home_team}-{away_team}-{match_date.split('T')[0]}"
                batch.set(matches_ref.document(doc_id.replace(" ", "_")), match_doc)
                matches_saved += 1
                    
            except Exception as e:
                logger.error(f"Error processing match: {e}")
                continue
        
        if matches_saved > 0:
            batch.commit()
        logger.info(f"[Startup] Saved {matches_saved} new finished matches to Firestore.")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"[Startup] Error fetching matches: {e}")
    except Exception as e:
        logger.error(f"[Startup] Error updating matches: {e}")



# Load all models (EPL and Financial) on application startup
@app.on_event("startup")
def load_all_models():
    """Combines loading logic for EPL and Financial ML models and updates Firestore data."""
    global PREDICTOR_MODEL, SCALER
    global FIN_MODELS, FIN_SCALERS
    
    logger.info("\n========== APPLICATION STARTUP ==========")
    
    # Get the Firestore client dependency manually for startup
    # Note: Using next(get_db()) works if init_db() was successful
    try:
        db_client = next(get_db())
    except Exception as e:
        logger.error(f"FATAL: Could not get Firestore client during startup: {e}")
        db_client = None

    if db_client:
        # 1. Update data from football API (Existing EPL Logic, now using Firestore client)
        api_token = os.getenv("FOOTBALL_DATA_API_TOKEN")
        update_league_standings(api_token, db_client)
        update_current_season_matches(api_token, db_client)
    else:
        logger.error("Skipping all database operations due to failed Firestore initialization.")

    # 2. Load EPL ML model and scaler (Unchanged)
    try:
        PREDICTOR_MODEL = joblib.load(MODEL_PATH)
        SCALER = joblib.load(SCALER_PATH)
        logger.info("EPL Machine Learning model and scaler loaded successfully.")
    except FileNotFoundError:
        logger.warning("EPL WARNING: Model files not found. Run train_model.py first.")
    except Exception as e:
        logger.error(f"EPL ERROR: Failed to load ML model: {e}")

    # 3. Load Financial Models for all top tickers (Unchanged)
    for ticker in TOP_TICKERS:
        model_path = os.path.join(FINANCE_MODEL_DIR, f'{ticker}_predictor_model.joblib')
        scaler_path = os.path.join(FINANCE_MODEL_DIR, f'{ticker}_predictor_scaler.joblib')
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                FIN_MODELS[ticker] = joblib.load(model_path)
                FIN_SCALERS[ticker] = joblib.load(scaler_path)
                logger.info(f"Financial model artifacts loaded for {ticker}.")
            else:
                logger.warning(f"Financial WARNING: Model files not found for {ticker}.")
        except Exception as e:
            logger.error(f"Financial ERROR: Failed to load model for {ticker}: {e}")
    
    logger.info("========== STARTUP COMPLETE ==========\n")


# --- Financial Helper Function (Unchanged) ---

def get_next_day_prediction(ticker: str, last_data: pd.DataFrame) -> Optional[float]:
    """
    Calculates features for the last row of data and predicts the next day's close price,
    using the specific model for the given ticker.
    """
    model = FIN_MODELS.get(ticker)
    scaler = FIN_SCALERS.get(ticker)
    
    if model is None or scaler is None or last_data.empty:
        return None
        
    features = ['Close', 'MA_7', 'MA_14', 'Volume', 'Volatility']
    
    if not all(f in last_data.columns for f in features):
        logger.error(f"Missing required features in data for prediction: {features}")
        return None

    latest_features = last_data[features].iloc[-1].values.reshape(1, -1)
    latest_features_scaled = scaler.transform(latest_features)
    
    prediction = model.predict(latest_features_scaled)[0]
    
    return round(float(prediction), 2)


# --- API Endpoints (MODIFIED FOR FIRESTORE) ---
@app.get("/api/epl", response_class=templates.TemplateResponse)
def serve_dashboard(request: Request):
    """Serves the main HTML dashboard template (EPL)."""
    return templates.TemplateResponse("epl.html", {"request": request})

@app.get("/api/finance", response_class=templates.TemplateResponse)
def serve_dashboard(request: Request):
    """Serves the main HTML dashboard template (EPL)."""
    return templates.TemplateResponse("finance.html", {"request": request})
# Existing EPL Endpoints
@app.get("/", response_class=templates.TemplateResponse)
def serve_dashboard(request: Request):
    """Serves the main HTML dashboard template (EPL)."""
    return templates.TemplateResponse("index.html", {"request": request})



@app.get("/api/current-table", response_model=List[Dict[str, Any]])
def get_current_table(db: Client = Depends(get_db)): # CHANGED dependency type to Client
    """
    Returns the current EPL table data from the Firestore.
    FIX: Retrieves the latest season ID and then queries the table.
    FIX: Removes Firestore's secondary order_by to avoid composite index requirement, sorting locally instead.
    """
    
    table_ref = db.collection(EPL_TABLE_COLLECTION)
    
    # 1. Find the current season ID by querying the most recently updated document
    latest_doc_query = table_ref.order_by("season", direction=Query.DESCENDING).limit(1).get()
   
    if not latest_doc_query:
        raise HTTPException(status_code=404, detail="No league standings found in Firestore.")
        
    # FIX 1: Safely extract the season ID from the first document
    first_doc_snapshot = latest_doc_query[0]
    
    if not first_doc_snapshot.exists:
        raise HTTPException(status_code=404, detail="Latest league standing document not found.")
        
    current_season_id = first_doc_snapshot.get('season') 

    if current_season_id is None:
        raise HTTPException(status_code=404, detail="Could not determine current season ID from Firestore data.")
    
    # Ensure current_season_id matches the data type in the database (assuming INT based on update logic)
    try:
        current_season_id = int(current_season_id)
    except ValueError:
        raise HTTPException(status_code=500, detail="Database 'season' field has wrong format for integer casting.")

    # 2. Query table data only by the season ID (FIX 2: Removing .order_by("position"))
    # This prevents the composite index requirement crash.
    table_data_query = table_ref.where("season", "==", current_season_id).get()
    
    # Convert query results to a list of dictionaries
    table_data = [doc.to_dict() for doc in table_data_query]
    
    if not table_data:
        raise HTTPException(status_code=404, detail=f"League standings not found for season {current_season_id}.")

    # 3. Sort the data locally by position (Python handles this sort efficiently)
    # Use 99 as a fallback position if 'position' field is missing, ensuring those records sink to the bottom.
    table_data.sort(key=lambda t: t.get('position', 99))

    # 4. Format for JSON response
    response_data = [{
        "position": t.get('position'),
        "team": t.get('team'),
        "played": t.get('played'),
        "points": t.get('points'),
        "gd": t.get('goal_difference')
    } for t in table_data]
    
    return response_data


LABEL_ENCODER = joblib.load("models/predictor_label_encoder.joblib") 
@app.post("/api/predict", response_model=TeamPrediction)
def predict_match(
    home_team: str, 
    away_team: str, 
    features: TeamPredictionFeatures
):
    """Makes a prediction using the loaded EPL ML model."""
    global PREDICTOR_MODEL, SCALER
    
    if PREDICTOR_MODEL is None or SCALER is None:
        raise HTTPException(status_code=500, detail="Prediction model not loaded.")

    # Prepare input data
    input_df = pd.DataFrame([[features.home_pts_last_5, features.away_pts_last_5]],
                            columns=['home_pts_last_5', 'away_pts_last_5'])
    
    # Scale the input data using the saved scaler
    scaled_input = SCALER.transform(input_df)
    
    # Predict probabilities and class
    numeric_classes = PREDICTOR_MODEL.classes_  # e.g., [0,1,2]
    probabilities = PREDICTOR_MODEL.predict_proba(scaled_input)[0]
    predicted_numeric_class = PREDICTOR_MODEL.predict(scaled_input)[0]

    # Decode numeric labels to 'H','D','A'
    classes = LABEL_ENCODER.inverse_transform(numeric_classes)
    predicted_class = LABEL_ENCODER.inverse_transform([predicted_numeric_class])[0]

    # Map classes to human-readable descriptions
    result_map = {'H': f"{home_team} Win", 'D': "Draw", 'A': f"{away_team} Win"}

    prob_map = dict(zip(classes, probabilities))

    return TeamPrediction(
        home_team=home_team,
        away_team=away_team,
        probabilities={result_map[c]: prob_map.get(c, 0.0) for c in classes},
        prediction=result_map[predicted_class]
    )


# --- Financial Analytics Endpoints (Unchanged) ---

@app.get("/analytics", response_class=HTMLResponse)
async def serve_analytics_page(request: Request):
    """Serves the Financial Market Analytics HTML page."""
    try:
        with open("finance.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="financial_analytics.html not found.")

async def fetch_live_finance_data(ticker: str, period: str = "6mo", interval: str = "1d"):
    """Fetches historical daily data using yfinance."""
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data returned from yfinance for {ticker}.")
        return data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    except Exception as e:
        logger.error(f"Error fetching live data for {ticker}: {e}")
        return pd.DataFrame()


@app.get("/api/finance/data/{ticker}", response_model=Dict[str, Any])
async def get_finance_data(ticker: str):
    ticker = ticker.upper()

    # Load model/scaler if not already loaded
    model = FIN_MODELS.get(ticker)
    scaler = FIN_SCALERS.get(ticker)
    if model is None or scaler is None:
        try:
            model, scaler = load_single_model(ticker)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model not found or failed to load: {e}")

    # Fetch historical data
    historical_data = await fetch_live_finance_data(ticker, period="1y", interval="1d")
    if historical_data.empty:
        return {
            "ticker": ticker,
            "current_price": None,
            "next_day_forecast": None,
            "historical_data": [],
            "forecast_data": [],
            "correlation_matrix": {},
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    # Feature engineering
    df = historical_data.copy()
    df["MA_7"] = df["Close"].rolling(7).mean()
    df["MA_14"] = df["Close"].rolling(14).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Daily_Return"].rolling(14).std()
    df = df.dropna()
    if df.empty:
        raise HTTPException(status_code=400, detail="Not enough data for feature computation.")

    # Latest features for prediction
    features = ['Close', 'MA_7', 'MA_14', 'Volume', 'Volatility']
    latest_features = df[features].iloc[-1].values.reshape(1, -1)
    try:
        scaled_features = scaler.transform(latest_features)
        predicted_close = float(model.predict(scaled_features)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    last_close = float(df['Close'].iloc[-1])
    predicted_return = round(((predicted_close - last_close) / last_close) * 100, 4)

    # 7-day forecast (simple drift + noise)
    forecast_data = []
    current_price = last_close
    for i in range(1, 8):
        drift = (predicted_close - current_price) * 0.3
        current_price += drift + np.random.normal(0, 0.002 * last_close)
        forecast_data.append({
            "date": (datetime.utcnow() + timedelta(days=i)).strftime("%Y-%m-%d"),
            "price": round(float(current_price), 2)
        })

    # Correlation matrix
    corr_df = df[['Close', 'Volume', 'MA_7', 'MA_14']].corr().fillna(0)
    correlation_matrix = corr_df.applymap(lambda x: float(x)).to_dict()

    # Historical data (last 120 days)
    hist_df_reset = df[['Close', 'MA_7']].tail(120).reset_index()
    hist_df_reset.rename(columns={hist_df_reset.columns[0]: 'Date'}, inplace=True)
    hist_df_reset.columns = [str(c).replace(" ", "_") for c in hist_df_reset.columns]

    # Build historical records safely
    historical_records = [
        {
            "Date": row[0].strftime("%Y-%m-%d") if isinstance(row[0], pd.Timestamp) else str(row[0]),
            "Close": float(row[1]),
            "MA_7": float(row[2]) if not pd.isna(row[2]) else 0.0
        }
        for row in hist_df_reset.itertuples(index=False, name=None)
    ]

    # Return full response
    return {
        "ticker": ticker,
        "current_price": round(last_close, 2),
        "next_day_forecast": round(predicted_close, 2),
        "predicted_return_percent": predicted_return,
        "historical_data": historical_records,
        "forecast_data": forecast_data,
        "correlation_matrix": correlation_matrix,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }




# Helper function (Unchanged)
def load_single_model(ticker: str):
    """Loads a single ML model and scaler artifact into the global dictionaries."""
    try:
        model_path = os.path.join(MODEL_DIR, f"{ticker}_predictor_model.joblib")
        scaler_path = os.path.join(MODEL_DIR, f"{ticker}_predictor_scaler.joblib")

        print("Looking for model at:", model_path)

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise HTTPException(
                status_code=404,
                detail=f"Prediction model not found for ticker: {ticker}. Please run the training script for this ticker."
            )

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        FIN_MODELS[ticker] = model
        FIN_SCALERS[ticker] = scaler

        print(f"[Startup] Successfully loaded finance model for {ticker}.")
        return model, scaler

    except Exception as e:
        print(f"[Startup] ERROR: Failed to load finance model for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

  
@app.get("/analytics")
def predict(ticker: str):
    try:
        model, scaler = load_single_model(ticker)
        data = yf.download(ticker, period="6mo", interval="1d")
        last_features = data.tail(10)[["Open", "High", "Low", "Close", "Volume"]]

        # Scale features
        scaled = scaler.transform(last_features)

        # Forecast next value
        prediction = model.predict(scaled[-1].reshape(1, -1))[0]

        # Compute correlation matrix safely
        correlation_matrix = data.corr().to_dict()

        # Prepare historical data for Chart.js
        historical_data = [
            {"date": str(idx.date()), "close": float(row["Close"])}
            for idx, row in data.tail(60).iterrows()
        ]

        return {
            "ticker": ticker,
            "current_price": float(data["Close"].iloc[-1]),
            "next_day_forecast": float(prediction),
            "correlation_matrix": correlation_matrix,
            "historical_data": historical_data,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
      
        
  

# --- Run Command ---
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)