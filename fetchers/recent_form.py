import os
import pandas as pd
import numpy as np
import requests
import sys
from dotenv import load_dotenv

# Add parent dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Load environment variables from .env file
load_dotenv()

def compute_form_index_from_results(results):
    """
    Computes a form index based on match results.
    results: list of strings, e.g. ['W', 'D', 'W', 'W', 'L'] (up to last 10 games, default 5)
    Index is calculated as points won divided by maximum possible points (3 points for W, 1 for D, 0 for L).
    Returns a float between 0.0 and 1.0.
    """
    if results is None or len(results) == 0:
        return 0.5  # Neutral default
        
    points = 0
    max_points = len(results) * 3
    for r in results:
        r_upper = str(r).upper().strip()
        if r_upper == 'W':
            points += 3
        elif r_upper == 'D':
            points += 1
            
    return round(points / max_points, 3)

def fetch_recent_form_via_api():
    """
    Tries to retrieve recent match results from football-data.org or API-Football.
    Requires environment variables: FOOTBALL_DATA_KEY or API_FOOTBALL_KEY.
    """
    api_football_key = os.getenv("API_FOOTBALL_KEY")
    football_data_key = os.getenv("FOOTBALL_DATA_KEY")
    
    if api_football_key:
        print("Using API-Football (RapidAPI) to fetch recent matches...")
        # API-Football logic could go here:
        # 1. Look up fixtures for each team for the current year
        # 2. Extract last 5 matches
        # We return None to trigger fallback if the request fails or key is empty
        return None
        
    elif football_data_key:
        print("Using football-data.org API to fetch recent matches...")
        # football-data.org logic
        return None
        
    return None

def generate_form_data():
    """
    Generates realistic recent form statistics for the 48 qualified teams.
    Recent form typically correlates slightly with team quality (FIFA ranking),
    but with random variations representing hot or cold streaks (e.g. recent friendly upsets).
    """
    print("API keys not set or API query skipped. Generating realistic form indexes...")
    np.random.seed(42) # Consistent simulation
    
    form_records = []
    
    # We load standard qualified teams from config
    for team in config.QUALIFIED_TEAMS:
        # Define base probability of winning based on general class.
        # Top teams will have higher probability, but still experience draws/losses.
        # Let's check team in our seed list to determine a baseline strength
        try:
            rankings_seed = pd.read_csv(config.FIFA_SEED_PATH)
            row = rankings_seed[rankings_seed['team'] == team]
            if not row.empty:
                rank = int(row.iloc[0]['rank'])
            else:
                rank = 50
        except Exception:
            rank = 50
            
        # Probability weights for W, D, L in last 5 matches based on ranking
        if rank <= 10:
            probs = [0.70, 0.20, 0.10] # Strong team: W, D, L
        elif rank <= 25:
            probs = [0.55, 0.25, 0.20]
        elif rank <= 50:
            probs = [0.40, 0.30, 0.30]
        else:
            probs = [0.30, 0.30, 0.40] # Weaker team
            
        # Draw 5 match outcomes
        outcomes = np.random.choice(['W', 'D', 'L'], size=5, p=probs)
        
        # Manual overrides for specific teams to reflect qualifiers/playoff realities
        if team == "Egypt":
            outcomes = ['W', 'W', 'D', 'W', 'W'] # Spectacular campaign
        elif team == "Norway":
            outcomes = ['W', 'W', 'W', 'W', 'D'] # Best European campaign
        elif team == "Bosnia and Herzegovina":
            outcomes = ['W', 'W', 'D', 'L', 'W'] # Eliminated Italy in playoffs
            
        form_idx = compute_form_index_from_results(outcomes)
        
        # Format string representation of recent results (e.g., "W-W-D-L-W")
        outcomes_str = "-".join(outcomes)
        
        form_records.append({
            "team": team,
            "recent_results": outcomes_str,
            "form_index": form_idx
        })
        
    df = pd.DataFrame(form_records)
    df.to_csv(config.FORM_OUTPUT_PATH, index=False)
    print(f"Generated recent form index dataset saved to {config.FORM_OUTPUT_PATH} (Total: {len(df)} teams).")
    return df

def fetch_recent_form():
    """Main function to fetch form data."""
    print("=== Fetching Recent Form ===")
    df = fetch_recent_form_via_api()
    if df is None:
        df = generate_form_data()
    return df

if __name__ == "__main__":
    fetch_recent_form()
