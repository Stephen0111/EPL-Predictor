
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# --- Internal Imports ---
from database import init_db, get_db, SessionLocal, EPLMatch, EPLTable 

# --- Setup ---

app = FastAPI(title="EPL Predictor API")
# Initialize database on startup (or check if it exists)
init_db()

# Configure Templates for serving the frontend HTML
templates = Jinja2Templates(directory="templates")

# Configure CORS (Crucial for frontend fetching data from the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/predictions", StaticFiles(directory="predictions"), name="predictions")

# --- Pydantic Schemas ---

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

# --- Model Loading ---

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'predictor_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'predictor_scaler.joblib')

PREDICTOR_MODEL = None
SCALER = None

# --- Auto-Update Functions (Run on Startup) ---

def update_league_standings(api_token: str):
    """
    Fetches the current EPL standings from football-data.org and updates the EPLTable.
    Runs on application startup to ensure fresh data.
    """
    db = SessionLocal()
    
    print("[Startup] Updating current league standings...")
    
    if not api_token:
        print("Warning: FOOTBALL_DATA_API_TOKEN not set. Skipping standings update.")
        db.close()
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
            
            # Clear old standings and refresh with new data
            db.query(EPLTable).delete()
            
            for table in standings_data['standings']:
                if table['type'] == 'TOTAL': 
                    for team_info in table.get('table', []):
                        standing_obj = EPLTable(
                            season=current_season_id,
                            position=team_info.get('position'),
                            team=team_info['team']['name'],
                            played=team_info.get('playedGames'),
                            points=team_info.get('points'),
                            goal_difference=team_info.get('goalDifference'),
                        )
                        db.add(standing_obj)

            db.commit()
            print(f"[Startup] Successfully updated {db.query(EPLTable).count()} league standings.")
            return current_season_id

    except requests.exceptions.RequestException as e:
        db.rollback()
        print(f"[Startup] Error fetching standings: {e}")
    except Exception as e:
        db.rollback()
        print(f"[Startup] Error updating standings: {e}")
    finally:
        db.close()
    
    return None


def update_current_season_matches(api_token: str):
    """
    Fetches finished current season matches from football-data.org
    and updates the EPLMatch table. Runs on application startup.
    """
    db = SessionLocal()
    
    print("[Startup] Updating current season matches...")
    
    if not api_token:
        print("Warning: FOOTBALL_DATA_API_TOKEN not set. Skipping matches update.")
        db.close()
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
        
        for match in matches_data.get('matches', []):
            try:
                # Only process finished matches
                if match['status'] != 'FINISHED':
                    continue
                
                season = match['season']['id']
                home_team = match['homeTeam']['name']
                away_team = match['awayTeam']['name']
                home_goals = match['score']['fullTime']['home']
                away_goals = match['score']['fullTime']['away']
                match_date = match['utcDate']
                
                # Determine result
                result = 'H' if home_goals > away_goals else ('A' if home_goals < away_goals else 'D')
                
                # Check if match already exists
                existing = db.query(EPLMatch).filter(
                    EPLMatch.home_team == home_team,
                    EPLMatch.away_team == away_team,
                    EPLMatch.match_date == match_date
                ).first()
                
                if not existing:
                    # Get current team points from EPLTable
                    home_standing = db.query(EPLTable).filter(EPLTable.team == home_team).first()
                    away_standing = db.query(EPLTable).filter(EPLTable.team == away_team).first()
                    
                    home_pts = home_standing.points if home_standing else 0
                    away_pts = away_standing.points if away_standing else 0
                    
                    match_obj = EPLMatch(
                        season=season,
                        match_date=match_date,
                        home_team=home_team,
                        away_team=away_team,
                        full_time_home_goals=home_goals,
                        full_time_away_goals=away_goals,
                        result=result,
                        home_pts_last_5=home_pts,
                        away_pts_last_5=away_pts
                    )
                    db.add(match_obj)
                    matches_saved += 1
                    
            except Exception as e:
                continue
        
        db.commit()
        print(f"[Startup] Saved {matches_saved} new finished matches.")
    
    except requests.exceptions.RequestException as e:
        db.rollback()
        print(f"[Startup] Error fetching matches: {e}")
    finally:
        db.close()


# Load the model and scaler on application startup
@app.on_event("startup")
def load_model():
    global PREDICTOR_MODEL, SCALER
    
    print("\n========== APPLICATION STARTUP ==========")
    
    # 1. Update data from football API
    api_token = os.getenv("FOOTBALL_DATA_API_TOKEN")
    update_league_standings(api_token)
    update_current_season_matches(api_token)
    
    # 2. Load ML model and scaler
    try:
        PREDICTOR_MODEL = joblib.load(MODEL_PATH)
        SCALER = joblib.load(SCALER_PATH)
        print("[Startup] Machine Learning model and scaler loaded successfully.")
    except FileNotFoundError:
        print("[Startup] WARNING: Model files not found. Run train_model.py first.")
    except Exception as e:
        print(f"[Startup] ERROR: Failed to load ML model: {e}")
    
    print("========== STARTUP COMPLETE ==========\n")

# --- API Endpoints ---

@app.get("/", response_class=templates.TemplateResponse)
def serve_dashboard(request: Request):
    """Serves the main HTML dashboard template."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/current-table", response_model=List[Dict[str, Any]])
def get_current_table(db: Session = Depends(get_db)):
    """
    Returns the current EPL table data from the database.
    This data is automatically updated on application startup.
    """
    
    # Find the current season
    latest_entry = db.query(EPLTable.season).order_by(EPLTable.season.desc()).first()
    current_season_id = latest_entry[0] if latest_entry else None

    if not current_season_id:
        raise HTTPException(status_code=404, detail="No league standings found in database. Data will be refreshed on next application restart.")

    # Query the actual table data for the latest season
    table_data = db.query(EPLTable).filter(EPLTable.season == current_season_id).order_by(EPLTable.position).all()
    
    # If the table is empty, return a 404
    if not table_data:
        raise HTTPException(status_code=404, detail=f"League standings not found for season {current_season_id}.")

    # Format for JSON response
    response_data = [{
        "position": t.position,
        "team": t.team,
        "played": t.played,
        "points": t.points,
        "gd": t.goal_difference
    } for t in table_data]
    
    return response_data

@app.post("/api/predict", response_model=TeamPrediction)
def predict_match(
    home_team: str, 
    away_team: str, 
    features: TeamPredictionFeatures
):
    """Makes a prediction using the loaded ML model."""
    global PREDICTOR_MODEL, SCALER
    
    if PREDICTOR_MODEL is None or SCALER is None:
        raise HTTPException(status_code=500, detail="Prediction model not loaded.")

    # Prepare input data
    input_df = pd.DataFrame([[features.home_pts_last_5, features.away_pts_last_5]],
                            columns=['home_pts_last_5', 'away_pts_last_5'])
    
    # Scale the input data using the saved scaler
    scaled_input = SCALER.transform(input_df)
    
    # Get prediction probabilities
    probabilities = PREDICTOR_MODEL.predict_proba(scaled_input)[0]
    classes = PREDICTOR_MODEL.classes_ # [A, D, H]
    
    prob_map = dict(zip(classes, probabilities))
    
    # Determine the final prediction
    predicted_class = PREDICTOR_MODEL.predict(scaled_input)[0]
    
    # Map classes to human-readable format
    result_map = {'H': f"{home_team} Win", 'D': "Draw", 'A': f"{away_team} Win"}
    
    return TeamPrediction(
        home_team=home_team,
        away_team=away_team,
        probabilities={result_map[c]: prob_map.get(c, 0.0) for c in classes},
        prediction=result_map[predicted_class]
    )

# --- Run Command ---
# To start the server: uvicorn main:app --reload