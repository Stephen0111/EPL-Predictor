
import joblib
import pandas as pd
import numpy as np
import requests
import os
from dotenv import load_dotenv
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from database import init_db, SessionLocal, EPLMatch, EPLTable
from sqlalchemy.orm import Session


load_dotenv()

# Ensure DB is initialized (and tables are created)
init_db()

# --- 1. Data Ingestion from Kaggle (Historical Data) ---

def fetch_kaggle_historical_data(db: Session):
    """Fetches EPL historical data from Kaggle and saves it to the database."""
    print("Fetching historical data from Kaggle...")
    
    # First, ensure Kaggle API is configured
    kaggle_config = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_config):
        print("Error: Kaggle API key not found. Please configure it:")
        print("1. Download your kaggle.json from https://www.kaggle.com/settings/account")
        print("2. Place it in ~/.kaggle/kaggle.json")
        print("3. Run: chmod 600 ~/.kaggle/kaggle.json")
        return
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Kaggle API not installed. Run: pip install kaggle")
        return
    
    api = KaggleApi()
    api.authenticate()
    
    # Download the EPL dataset
    dataset_name = "evangower/english-premier-league-standings"
    download_path = "./kaggle_data"
    os.makedirs(download_path, exist_ok=True)
    
    print(f"Downloading dataset: {dataset_name}")
    try:
        api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        return

    # Parse the CSV files
    csv_files = [f for f in os.listdir(download_path) if f.endswith('.csv')]
    
    matches_from_kaggle = []
    
    for csv_file in csv_files:
        file_path = os.path.join(download_path, csv_file)
        print(f"Processing {csv_file}...")
        
        df = pd.read_csv(file_path)
        
        # Handle different CSV structures
        if 'Season' in df.columns and 'HomeTeam' in df.columns:
            # Process matches
            for _, row in df.iterrows():
                try:
                    # Type casting for safety
                    season = int(row['Season']) if 'Season' in row and pd.notna(row['Season']) else None
                    home_team = row.get('HomeTeam', 'Unknown')
                    away_team = row.get('AwayTeam', 'Unknown')
                    home_goals = int(row.get('FTHG', 0)) if pd.notna(row.get('FTHG')) else 0
                    away_goals = int(row.get('FTAG', 0)) if pd.notna(row.get('FTAG')) else 0
                    match_date = row.get('Date', str(datetime.now().date()))
                    
                    # Determine result
                    if home_goals > away_goals:
                        result = 'H'
                    elif home_goals < away_goals:
                        result = 'A'
                    else:
                        result = 'D'
                    
                    # Check if match already exists
                    existing = db.query(EPLMatch).filter(
                        EPLMatch.home_team == home_team,
                        EPLMatch.away_team == away_team,
                        EPLMatch.match_date == match_date
                    ).first()
                    
                    if not existing:
                        match = EPLMatch(
                            season=season,
                            match_date=match_date,
                            home_team=home_team,
                            away_team=away_team,
                            full_time_home_goals=home_goals,
                            full_time_away_goals=away_goals,
                            result=result,
                            home_pts_last_5=0,  # Will be calculated below
                            away_pts_last_5=0
                        )
                        db.add(match)
                        matches_from_kaggle.append({
                            'season': season,
                            'match_date': match_date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_goals': home_goals,
                            'away_goals': away_goals,
                            'result': result
                        })
                except Exception as e:
                    continue
            
            db.commit()
            print(f"Saved matches from {csv_file}")
    
    print("Kaggle historical data import complete.")
    return matches_from_kaggle


def calculate_team_form(matches: list, team_name: str, match_index: int, lookback: int = 5) -> int:
    """
    Calculates points for a team based on their last N matches before a given match.
    Returns the points accumulated in the lookback window.
    """
    points = 0
    matches_counted = 0
    
    # Look back from the current match
    for i in range(match_index - 1, -1, -1):
        if matches_counted >= lookback:
            break
        
        match = matches[i]
        
        if match['home_team'] == team_name:
            if match['result'] == 'H':
                points += 3
            elif match['result'] == 'D':
                points += 1
            matches_counted += 1
        elif match['away_team'] == team_name:
            if match['result'] == 'A':
                points += 3
            elif match['result'] == 'D':
                points += 1
            matches_counted += 1
    
    return points


def process_historical_data_for_training(db: Session, matches_from_kaggle: list):
    """
    Processes historical Kaggle data and updates EPLMatch with calculated form data.
    """
    print("\n[Training Data] Calculating team form from historical matches...")
    
    # Sort matches by date
    sorted_matches = sorted(matches_from_kaggle, key=lambda x: x['match_date'])
    
    # For each match, calculate the form (last 5 matches points) for both teams
    for idx, match in enumerate(sorted_matches):
        home_form = calculate_team_form(sorted_matches, match['home_team'], idx, lookback=5)
        away_form = calculate_team_form(sorted_matches, match['away_team'], idx, lookback=5)
        
        # Update the EPLMatch record with calculated form
        epl_match = db.query(EPLMatch).filter(
            EPLMatch.home_team == match['home_team'],
            EPLMatch.away_team == match['away_team'],
            EPLMatch.match_date == match['match_date']
        ).first()
        
        if epl_match:
            epl_match.home_pts_last_5 = home_form
            epl_match.away_pts_last_5 = away_form
    
    db.commit()
    print(f"Calculated form data for {len(sorted_matches)} historical matches.")


# ---------------------------------------------------------------------------------
# NEW FUNCTION: For the Dashboard Table (EPLTable)
# ---------------------------------------------------------------------------------

def update_league_standings(db: Session, api_token: str):
    """
    Fetches the current EPL standings from football-data.org and updates the EPLTable.
    This is for fast dashboard display (API: /api/current-table).
    """
    print("\n[Dashboard Data] Updating current league standings in EPLTable...")
    
    if not api_token:
        print("Error: FOOTBALL_DATA_API_TOKEN not set.")
        return None
    
    headers = {"X-Auth-Token": api_token}
    base_url = "https://api.football-data.org/v4"
    
    try:
        standings_url = f"{base_url}/competitions/PL/standings"
        standings_response = requests.get(standings_url, headers=headers)
        standings_response.raise_for_status()
        standings_data = standings_response.json()
        
        if 'standings' in standings_data:
            current_season_id = standings_data.get('season', {}).get('id')
            
            # CRITICAL: Clear the old standings data for a clean update
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
            print(f"Successfully saved {db.query(EPLTable).count()} league standings to EPLTable.")
            return current_season_id

    except requests.exceptions.RequestException as e:
        db.rollback()
        print(f"Error fetching standings from football-data.org: {e}")
    except Exception as e:
        db.rollback()
        print(f"An unexpected error occurred during standings update: {e}")
    
    return None

# ---------------------------------------------------------------------------------
# MODIFIED FUNCTION: For Model Training Data (EPLMatch)
# ---------------------------------------------------------------------------------

def get_team_points_from_db(team_name: str, db: Session) -> int:
    """Helper to retrieve team's total points from the locally cached EPLTable."""
    standing = db.query(EPLTable).filter(EPLTable.team == team_name).first()
    return standing.points if standing else 0


def fetch_current_season_data(db: Session, api_token: str, current_season: int = None):
    """
    Fetches finished current season matches from football-data.org,
    and retrieves team's *total* points from the local EPLTable for features.
    """
    print("\n[Model Data] Fetching current season matches for EPLMatch table...")
    
    if not api_token:
        print("Error: FOOTBALL_DATA_API_TOKEN not set.")
        return
    
    headers = {"X-Auth-Token": api_token}
    base_url = "https://api.football-data.org/v4"
    
    try:
        # Fetch current season matches
        matches_response = requests.get(
            f"{base_url}/competitions/PL/matches",
            headers=headers
        )
        matches_response.raise_for_status()
        matches_data = matches_response.json()
        
        matches_saved = 0
        
        for match in matches_data.get('matches', []):
            try:
                season = match['season']['id']
                if not current_season: current_season = season

                # Only process finished matches
                if match['status'] != 'FINISHED':
                    continue
                
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
                    # Look up points from the local EPLTable
                    home_pts = get_team_points_from_db(home_team, db)
                    away_pts = get_team_points_from_db(away_team, db)
                    
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
        print(f"Saved {matches_saved} new finished matches to EPLMatch.")
    
    except requests.exceptions.RequestException as e:
        db.rollback()
        print(f"Error fetching matches from football-data.org: {e}")


def predict_end_of_season_standings(db: Session, api_token: str):
    """
    Predicts the final EPL standings at the end of the season by simulating 
    remaining matches and using the trained model.
    """
    print("\n[Predictions] Generating end-of-season standings predictions...")
    
    if not api_token:
        print("Error: FOOTBALL_DATA_API_TOKEN not set.")
        return
    
    headers = {"X-Auth-Token": api_token}
    base_url = "https://api.football-data.org/v4"
    
    try:
        # Load the trained model and scaler
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        model = joblib.load(os.path.join(model_dir, 'predictor_model.joblib'))
        scaler = joblib.load(os.path.join(model_dir, 'predictor_scaler.joblib'))
        
        # Get current standings
        current_standings = db.query(EPLTable).all()
        if not current_standings:
            print("No current standings found. Run update_league_standings first.")
            return
        
        # Create a dict of teams and their current points
        team_points = {t.team: t.points for t in current_standings}
        team_games_played = {t.team: t.played for t in current_standings}
        
        # Fetch remaining fixtures
        fixtures_response = requests.get(
            f"{base_url}/competitions/PL/matches?status=SCHEDULED",
            headers=headers
        )
        fixtures_response.raise_for_status()
        fixtures_data = fixtures_response.json()
        
        remaining_matches = []
        for match in fixtures_data.get('matches', []):
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            remaining_matches.append((home_team, away_team))
        
        if not remaining_matches:
            print("No remaining fixtures found. Season may be complete.")
            return
        
        print(f"Found {len(remaining_matches)} remaining matches to simulate...")
        
        # Simulate remaining matches
        for home_team, away_team in remaining_matches:
            if home_team not in team_points or away_team not in team_points:
                continue
            
            home_pts = team_points[home_team]
            away_pts = team_points[away_team]
            
            # Prepare features for prediction
            features_df = pd.DataFrame([[home_pts, away_pts]], 
                                       columns=['home_pts_last_5', 'away_pts_last_5'])
            scaled_features = scaler.transform(features_df)
            
            # Predict match outcome
            prediction = model.predict(scaled_features)[0]
            
            # Update points based on prediction
            if prediction == 'H':
                team_points[home_team] += 3
            elif prediction == 'A':
                team_points[away_team] += 3
            else:  # Draw
                team_points[home_team] += 1
                team_points[away_team] += 1
            
            team_games_played[home_team] += 1
            team_games_played[away_team] += 1
        
        # Create predicted standings
        predicted_standings = []
        for team, points in sorted(team_points.items(), key=lambda x: x[1], reverse=True):
            predicted_standings.append({
                'team': team,
                'predicted_points': points,
                'games_played': team_games_played[team]
            })
        
        # Save predictions to CSV for dashboard display
        predictions_dir = os.path.join(os.path.dirname(__file__), 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        
        predictions_df = pd.DataFrame(predicted_standings)
        predictions_path = os.path.join(predictions_dir, 'epl_end_of_season_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        
        print(f"End-of-season predictions saved to: {predictions_path}")
        print("\nPredicted Top 5 Teams:")
        for i, pred in enumerate(predicted_standings[:5], 1):
            print(f"{i}. {pred['team']} - {pred['predicted_points']} points")
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching remaining fixtures: {e}")
    except Exception as e:
        print(f"Error generating predictions: {e}")


# --- 3. Model Training ---

def train_prediction_model():
    """Loads data from DB, trains a model, and saves it."""
    db = SessionLocal()
    api_token = os.getenv("FOOTBALL_DATA_API_TOKEN")
    
    print("\n--- Starting Data Ingestion and Model Training ---")

    # 1. Fetch and process historical data from Kaggle
    matches_from_kaggle = fetch_kaggle_historical_data(db)
    
    # 2. Calculate team form for historical matches
    if matches_from_kaggle:
        process_historical_data_for_training(db, matches_from_kaggle)
    
    # 3. Update League Table STANDINGS (EPLTable) for the dashboard
    current_season_id = update_league_standings(db, api_token) 
    
    # 4. Fetch current season MATCHES (EPLMatch)
    fetch_current_season_data(db, api_token, current_season=current_season_id)
    
    # 5. Generate end-of-season predictions
    predict_end_of_season_standings(db, api_token)
    
    # Load ALL data (historical + current) from DB for training
    print("\n[Model Training] Loading all match data from database...")
    matches = db.query(EPLMatch).all()
    data = [{
        'home_pts_last_5': m.home_pts_last_5,
        'away_pts_last_5': m.away_pts_last_5,
        'result': m.result
    } for m in matches]
    
    df = pd.DataFrame(data)
    
    if len(df) < 10:
        print("Not enough data to train. Exiting.")
        return
    
    print(f"Training on {len(df)} total matches (historical + current)...")
    
    # Define Features (X) and Target (Y)
    X = df[['home_pts_last_5', 'away_pts_last_5']]
    Y = df['result']  # H, D, A
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    # Scaling Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model: Logistic Regression for multi-class classification
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_train_scaled, Y_train)
    
    # Evaluate model
    train_score = model.score(X_train_scaled, Y_train)
    test_score = model.score(X_test_scaled, Y_test)
    print(f"\nModel Performance:")
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Save Model and Scaler
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'predictor_model.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'predictor_scaler.joblib'))
    
    print(f"\nModel training complete.")
    print(f"Model saved to: {model_dir}/predictor_model.joblib")
    print(f"Scaler saved to: {model_dir}/predictor_scaler.joblib")
    
    db.close()

if __name__ == '__main__':
    train_prediction_model()