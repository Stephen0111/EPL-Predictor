
import joblib
import pandas as pd
import numpy as np
import requests
import os
from dotenv import load_dotenv
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from google.cloud.firestore import Client
from database import init_db, get_db, EPL_MATCHES_COLLECTION, EPL_TABLE_COLLECTION

load_dotenv()

# Initialize Firestore client
try:
    DB_CLIENT = next(get_db())
except Exception as e:
    print(f"FATAL: Could not initialize Firestore client. Error: {e}")
    DB_CLIENT = None

if DB_CLIENT is None:
    exit(1)

# -------------------------
# 1. Historical Data
# -------------------------
def fetch_kaggle_historical_data(db: Client):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Kaggle API not installed. Run: pip install kaggle")
        return []
    
    api = KaggleApi()
    api.authenticate()
    
    dataset_name = "evangower/english-premier-league-standings"
    download_path = "./kaggle_data"
    os.makedirs(download_path, exist_ok=True)
    
    try:
        api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        return []

    csv_files = [f for f in os.listdir(download_path) if f.endswith('.csv')]
    
    matches_ref = db.collection(EPL_MATCHES_COLLECTION)
    batch = db.batch()
    
    # Clear old matches
    for doc in matches_ref.stream():
        batch.delete(doc.reference)
    
    match_counter = 0
    matches_from_kaggle = []

    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(download_path, csv_file))
        if 'Season' not in df.columns or 'HomeTeam' not in df.columns:
            continue

        for index, row in df.iterrows():
            try:
                season = int(row['Season']) if pd.notna(row['Season']) else None
                home_team = row.get('HomeTeam', 'Unknown')
                away_team = row.get('AwayTeam', 'Unknown')
                home_goals = int(row.get('FTHG', 0)) if pd.notna(row.get('FTHG')) else 0
                away_goals = int(row.get('FTAG', 0)) if pd.notna(row.get('FTAG')) else 0
                match_date = row.get('Date', str(datetime.now().date()))
                result = 'H' if home_goals > away_goals else ('A' if home_goals < away_goals else 'D')
                
                doc_id = f"{season}-{match_date}-{home_team}-{away_team}".replace(" ", "_")
                match_doc = {
                    'season': season,
                    'match_date': match_date,
                    'home_team': home_team,
                    'away_team': away_team,
                    'full_time_home_goals': home_goals,
                    'full_time_away_goals': away_goals,
                    'result': result,
                    'home_pts_last_5': 0,
                    'away_pts_last_5': 0
                }
                batch.set(matches_ref.document(doc_id), match_doc)
                matches_from_kaggle.append(match_doc)
                match_counter += 1

                if match_counter % 500 == 0:
                    batch.commit()
                    batch = db.batch()
            except:
                continue
    
    batch.commit()
    print(f"Kaggle historical data import complete. Total matches saved: {match_counter}.")
    return matches_from_kaggle


def calculate_team_form(matches: list, team_name: str, match_index: int, lookback: int = 5) -> int:
    points = 0
    matches_counted = 0
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


def process_historical_data_for_training(db: Client, matches_from_kaggle: list):
    sorted_matches = sorted(matches_from_kaggle, key=lambda x: x['match_date'])
    matches_ref = db.collection(EPL_MATCHES_COLLECTION)
    batch = db.batch()
    update_counter = 0
    for idx, match in enumerate(sorted_matches):
        home_form = calculate_team_form(sorted_matches, match['home_team'], idx)
        away_form = calculate_team_form(sorted_matches, match['away_team'], idx)
        doc_id = f"{match['season']}-{match['match_date']}-{match['home_team']}-{match['away_team']}".replace(" ", "_")
        batch.update(matches_ref.document(doc_id), {'home_pts_last_5': home_form, 'away_pts_last_5': away_form})
        update_counter += 1
        if update_counter % 500 == 0:
            batch.commit()
            batch = db.batch()
    batch.commit()
    print(f"Updated form data for {update_counter} historical matches.")


# -------------------------
# 2. League Table
# -------------------------
def update_league_standings(db: Client, api_token: str):
    if not api_token:
        print("FOOTBALL_DATA_API_TOKEN not set.")
        return None
    headers = {"X-Auth-Token": api_token}
    base_url = "https://api.football-data.org/v4"
    table_ref = db.collection(EPL_TABLE_COLLECTION)
    
    try:
        resp = requests.get(f"{base_url}/competitions/PL/standings", headers=headers)
        resp.raise_for_status()
        data = resp.json()
        current_season_id = data.get('season', {}).get('id')
        batch = db.batch()
        for doc in table_ref.stream(): batch.delete(doc.reference)
        team_count = 0
        for table in data.get('standings', []):
            if table['type'] != 'TOTAL': continue
            for t in table.get('table', []):
                doc_ref = table_ref.document(t['team']['name'].replace(" ", "_"))
                batch.set(doc_ref, {
                    "season": current_season_id,
                    "position": t.get('position'),
                    "team": t['team']['name'],
                    "played": t.get('playedGames'),
                    "points": t.get('points'),
                    "goal_difference": t.get('goalDifference')
                })
                team_count += 1
        batch.commit()
        print(f"Saved {team_count} league standings.")
        return current_season_id
    except Exception as e:
        print(f"Error fetching standings: {e}")
        return None


def get_team_points_from_db(team_name: str, db: Client) -> int:
    doc = db.collection(EPL_TABLE_COLLECTION).document(team_name.replace(" ", "_")).get()
    return doc.get('points') if doc.exists else 0


def fetch_current_season_data(db: Client, api_token: str, current_season: int = None):
    if not api_token: return
    headers = {"X-Auth-Token": api_token}
    base_url = "https://api.football-data.org/v4"
    matches_ref = db.collection(EPL_MATCHES_COLLECTION)
    try:
        resp = requests.get(f"{base_url}/competitions/PL/matches", headers=headers)
        resp.raise_for_status()
        matches_saved = 0
        batch = db.batch()
        for match in resp.json().get('matches', []):
            if match['status'] != 'FINISHED': continue
            season = match['season']['id']
            if not current_season: current_season = season
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            home_goals = match['score']['fullTime']['home']
            away_goals = match['score']['fullTime']['away']
            match_date = match['utcDate']
            result = 'H' if home_goals > away_goals else ('A' if home_goals < away_goals else 'D')
            doc_id = f"{season}-{match_date}-{home_team}-{away_team}".replace(" ", "_")
            if not matches_ref.document(doc_id).get().exists:
                home_pts = get_team_points_from_db(home_team, db)
                away_pts = get_team_points_from_db(away_team, db)
                batch.set(matches_ref.document(doc_id), {
                    'season': season,
                    'match_date': match_date,
                    'home_team': home_team,
                    'away_team': away_team,
                    'full_time_home_goals': home_goals,
                    'full_time_away_goals': away_goals,
                    'result': result,
                    'home_pts_last_5': home_pts,
                    'away_pts_last_5': away_pts
                })
                matches_saved += 1
        batch.commit()
        print(f"Saved {matches_saved} finished matches.")
    except Exception as e:
        print(f"Error fetching current season matches: {e}")


# -------------------------
# 3. End-of-Season Predictions
# -------------------------
def predict_end_of_season_standings(db: Client, api_token: str):
    if not api_token: return
    headers = {"X-Auth-Token": api_token}
    base_url = "https://api.football-data.org/v4"

    try:
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        model = joblib.load(os.path.join(model_dir, 'predictor_model.joblib'))
        scaler = joblib.load(os.path.join(model_dir, 'predictor_scaler.joblib'))
        le = joblib.load(os.path.join(model_dir, 'predictor_label_encoder.joblib'))

        table_ref = db.collection(EPL_TABLE_COLLECTION)
        standings_docs = table_ref.order_by("position").get()
        current_standings = [d.to_dict() for d in standings_docs]
        if not current_standings: return

        team_points = {t['team']: t['points'] for t in current_standings}
        team_games_played = {t['team']: t['played'] for t in current_standings}

        fixtures_resp = requests.get(f"{base_url}/competitions/PL/matches?status=SCHEDULED", headers=headers)
        fixtures_resp.raise_for_status()
        remaining_matches = [(m['homeTeam']['name'], m['awayTeam']['name']) for m in fixtures_resp.json().get('matches', [])]
        if not remaining_matches: return

        for home_team, away_team in remaining_matches:
            if home_team not in team_points or away_team not in team_points: continue
            features_df = pd.DataFrame([[team_points[home_team], team_points[away_team]]],
                                       columns=['home_pts_last_5', 'away_pts_last_5'])
            scaled_features = scaler.transform(features_df)
            pred = model.predict(scaled_features)[0]
            pred_label = le.inverse_transform([pred])[0]

            if pred_label == 'H': team_points[home_team] += 3
            elif pred_label == 'A': team_points[away_team] += 3
            else: team_points[home_team] += 1; team_points[away_team] += 1

            team_games_played[home_team] += 1
            team_games_played[away_team] += 1

        predicted_standings = [{'team': t, 'predicted_points': p, 'games_played': team_games_played[t]}
                               for t, p in sorted(team_points.items(), key=lambda x: x[1], reverse=True)]

        os.makedirs(os.path.join(os.path.dirname(__file__), 'predictions'), exist_ok=True)
        predictions_df = pd.DataFrame(predicted_standings)
        predictions_df.to_csv(os.path.join(os.path.dirname(__file__), 'predictions/epl_end_of_season_predictions.csv'), index=False)
        print("Predictions saved. Top 5 Teams:")
        for i, pred in enumerate(predicted_standings[:5], 1):
            print(f"{i}. {pred['team']} - {pred['predicted_points']} points")
    except Exception as e:
        print(f"Error generating predictions: {e}")


# -------------------------
# 4. Model Training
# -------------------------
def train_prediction_model(db: Client):
    api_token = os.getenv("FOOTBALL_DATA_API_TOKEN")
    print("\n--- Starting Data Ingestion and Model Training ---")
    matches_from_kaggle = fetch_kaggle_historical_data(db)
    if matches_from_kaggle:
        process_historical_data_for_training(db, matches_from_kaggle)
    current_season_id = update_league_standings(db, api_token)
    fetch_current_season_data(db, api_token, current_season=current_season_id)

    matches_docs = db.collection(EPL_MATCHES_COLLECTION).get()
    df = pd.DataFrame([{
        'home_pts_last_5': m.get('home_pts_last_5'),
        'away_pts_last_5': m.get('away_pts_last_5'),
        'result': m.get('result')
    } for m in matches_docs]).dropna()

    if len(df) < 10:
        print("Not enough data to train. Exiting.")
        return

    X = df[['home_pts_last_5', 'away_pts_last_5']]
    Y = df['result']

    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss',
                          use_label_encoder=False, n_estimators=1000, max_depth=4,
                          learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X_train_scaled, Y_train)

    print(f"\nTrain accuracy: {model.score(X_train_scaled, Y_train):.4f}")
    print(f"Test accuracy: {model.score(X_test_scaled, Y_test):.4f}")

    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'predictor_model.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'predictor_scaler.joblib'))
    joblib.dump(le, os.path.join(model_dir, 'predictor_label_encoder.joblib'))

    print("Model training complete. Saved model, scaler, and label encoder.")
    predict_end_of_season_standings(db, api_token)


if __name__ == '__main__':
    if DB_CLIENT:
        train_prediction_model(DB_CLIENT)
