import os
import pandas as pd
import json
import sys

# Add parent dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from fetchers.fifa_ranking import fetch_fifa_rankings
from fetchers.transfermarkt import fetch_squad_values
from fetchers.recent_form import fetch_recent_form

def normalize_team_name(name):
    """
    Normalizes a team name based on configurations in config.py
    to align names across diverse data sources.
    """
    if not name or pd.isna(name):
        return None
    name = str(name).strip()
    
    # Check mappings
    for standard_name, alt_names in config.TEAM_NAME_MAPPINGS.items():
        if name in alt_names or name.lower() == standard_name.lower():
            return standard_name
            
    # Try case-insensitive matching with qualified teams
    for standard in config.QUALIFIED_TEAMS:
        if name.lower() == standard.lower():
            return standard
            
    return name

def load_group_mapping():
    """Loads group mapping from world_cup_2026_groups.json and returns a team -> group dict."""
    if not os.path.exists(config.GROUPS_JSON_PATH):
        raise FileNotFoundError(f"Groups config file not found at {config.GROUPS_JSON_PATH}")
        
    with open(config.GROUPS_JSON_PATH, 'r') as f:
        groups_data = json.load(f)
        
    team_to_group = {}
    for group_name, teams in groups_data.items():
        for team in teams:
            team_to_group[team] = group_name
    return team_to_group

def prepare_dataset():
    """
    Main pipeline function that runs all fetchers, reads outputs,
    performs self-healing data alignment, and generates final_features.csv.
    """
    print("\n=== Running Dataset Preparation Pipeline ===")
    
    # 1. Fetch data (which saves CSVs to config output paths)
    fifa_df = fetch_fifa_rankings()
    squad_df = fetch_squad_values()
    form_df = fetch_recent_form()
    
    # Load Group Mappings
    team_to_group = load_group_mapping()
    
    # Load Seed Datasets for healing missing values
    fifa_seed = pd.read_csv(config.FIFA_SEED_PATH)
    squad_seed = pd.read_csv(config.SQUAD_SEED_PATH)
    
    # Load Metadata (FC 26 Ratings and Qualifiers PPG)
    meta_df = pd.read_csv(config.METADATA_SEED_PATH)
    meta_df.to_csv(config.METADATA_OUTPUT_PATH, index=False)
    
    # Normalize names in fetched data
    fifa_df['clean_team'] = fifa_df['team'].apply(normalize_team_name)
    squad_df['clean_team'] = squad_df['team'].apply(normalize_team_name)
    form_df['clean_team'] = form_df['team'].apply(normalize_team_name)
    
    # Clean duplicates in clean names
    fifa_df = fifa_df.dropna(subset=['clean_team']).drop_duplicates(subset=['clean_team'])
    squad_df = squad_df.dropna(subset=['clean_team']).drop_duplicates(subset=['clean_team'])
    form_df = form_df.dropna(subset=['clean_team']).drop_duplicates(subset=['clean_team'])
    
    # Create final aligned records for all 48 qualified teams
    final_records = []
    
    for team in config.QUALIFIED_TEAMS:
        # Get Group
        group = team_to_group.get(team, "Unknown")
        
        # 1. Extract FIFA Rank & Points
        fifa_match = fifa_df[fifa_df['clean_team'] == team]
        if not fifa_match.empty:
            rank = fifa_match.iloc[0]['rank']
            points = fifa_match.iloc[0]['points']
        else:
            # Self-healing from seed
            seed_match = fifa_seed[fifa_seed['team'] == team]
            rank = seed_match.iloc[0]['rank'] if not seed_match.empty else 50
            points = seed_match.iloc[0]['points'] if not seed_match.empty else 1450.0
            print(f"[Self-Healing] Missing FIFA ranking for {team}. Loaded from seed: Rank {rank}")
            
        # 2. Extract Squad Value
        squad_match = squad_df[squad_df['clean_team'] == team]
        if not squad_match.empty:
            squad_value = squad_match.iloc[0]['squad_value_m']
        else:
            # Self-healing from seed
            seed_match = squad_seed[squad_seed['team'] == team]
            squad_value = seed_match.iloc[0]['squad_value_m'] if not seed_match.empty else 20.0
            print(f"[Self-Healing] Missing Squad Value for {team}. Loaded from seed: {squad_value}M")
            
        # 3. Extract Recent Form
        form_match = form_df[form_df['clean_team'] == team]
        if not form_match.empty:
            recent_results = form_match.iloc[0]['recent_results']
            form_index = form_match.iloc[0]['form_index']
        else:
            recent_results = "W-D-L-W-D"
            form_index = 0.5
            print(f"[Self-Healing] Missing Recent Form for {team}. Using default.")
            
        # 4. Extract FC 26 Rating & Qualifiers PPG
        meta_match = meta_df[meta_df['team'] == team]
        if not meta_match.empty:
            fc26_rating = meta_match.iloc[0]['fc26_rating']
            qualifiers_ppg = meta_match.iloc[0]['qualifiers_ppg']
        else:
            fc26_rating = 75
            qualifiers_ppg = 1.5
            print(f"[Self-Healing] Missing Metadata for {team}. Using defaults: FC26 {fc26_rating}, PPG {qualifiers_ppg}")
            
        # Get Confederation and Multiplier for strength adjustments
        confed = config.CONFEDERATION_MAPPING.get(team, "UEFA")
        mult = config.CONFEDERATION_MULTIPLIERS.get(confed, 1.0)
        
        # Apply adjustments
        adj_points = float(points) * mult
        adj_squad_value = float(squad_value) * config.TEAM_VALUE_MULTIPLIERS.get(team, 1.0)
        adj_ppg = float(qualifiers_ppg) * mult
            
        final_records.append({
            "team": team,
            "group": group,
            "fifa_rank": int(rank), # Will be overwritten based on adjusted points
            "fifa_points": round(adj_points, 1),
            "squad_value_m": round(adj_squad_value, 2),
            "recent_results": recent_results,
            "recent_form_index": float(form_index),
            "fc26_rating": int(fc26_rating),
            "qualifiers_ppg": round(adj_ppg, 2)
        })
        
    final_df = pd.DataFrame(final_records)
    
    # Recalculate relative FIFA rankings among the 48 participants based on adjusted points
    final_df = final_df.sort_values(by="fifa_points", ascending=False)
    final_df["fifa_rank"] = range(1, len(final_df) + 1)
    
    # Sort alphabetically by team name to keep order consistent
    final_df = final_df.sort_values(by="team")
    
    final_df.to_csv(config.FINAL_DATASET_PATH, index=False)
    print(f"\nFinal adjusted and normalized dataset written to {config.FINAL_DATASET_PATH}.")
    print(f"Dataset summary: {len(final_df)} teams aligned with confederation-adjusted features.")
    
    return final_df

if __name__ == "__main__":
    prepare_dataset()
