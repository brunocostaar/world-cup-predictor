import os
import pandas as pd
import numpy as np
import pickle
import json
import sys

# Add parent dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Global cache variables initialized to None for lazy-loading
df_teams = None
team_features = None
rf_model = None
poisson_model = None
lambda_cache = {}

def load_prediction_models():
    """Lazy loads prediction models and team features from disk to avoid boot import errors."""
    global df_teams, team_features, rf_model, poisson_model
    if rf_model is None or poisson_model is None:
        if not os.path.exists(config.FINAL_DATASET_PATH):
            raise FileNotFoundError(f"Final features dataset not found at {config.FINAL_DATASET_PATH}. Please run data prep first.")
        df_teams = pd.read_csv(config.FINAL_DATASET_PATH)
        team_features = df_teams.set_index('team').to_dict(orient='index')
        
        rf_path = os.path.join(config.DATA_DIR, "rf_model.pkl")
        poisson_path = os.path.join(config.DATA_DIR, "poisson_model.pkl")
        
        if not os.path.exists(rf_path) or not os.path.exists(poisson_path):
            raise FileNotFoundError("Trained models not found. Please run model training (train.py) first.")
            
        with open(rf_path, 'rb') as f:
            rf_model = pickle.load(f)
        with open(poisson_path, 'rb') as f:
            poisson_model = pickle.load(f)

def get_match_lambda(team_A, team_B):
    """
    Calculates expected goals (lambdas) for team_A and team_B in a match.
    Uses a symmetric Poisson regression model predicting goals scored based on feature diffs.
    Uses a cache to avoid calling model.predict millions of times.
    """
    pair = (team_A, team_B)
    if pair in lambda_cache:
        return lambda_cache[pair]
        
    load_prediction_models()
    feat_A = team_features[team_A]
    feat_B = team_features[team_B]
    
    # 1. Team A goals (features: A - B)
    rank_diff_A = feat_B['fifa_rank'] - feat_A['fifa_rank']
    log_value_ratio_A = np.log10(feat_A['squad_value_m'] / (feat_B['squad_value_m'] + 1.0))
    form_diff_A = feat_A['recent_form_index'] - feat_B['recent_form_index']
    fc26_diff_A = feat_A['fc26_rating'] - feat_B['fc26_rating']
    qualifiers_ppg_diff_A = feat_A['qualifiers_ppg'] - feat_B['qualifiers_ppg']
    
    lambda_A = poisson_model.predict([[rank_diff_A, log_value_ratio_A, form_diff_A, fc26_diff_A, qualifiers_ppg_diff_A]])[0]
    
    # 2. Team B goals (features: B - A)
    rank_diff_B = feat_A['fifa_rank'] - feat_B['fifa_rank']
    log_value_ratio_B = np.log10(feat_B['squad_value_m'] / (feat_A['squad_value_m'] + 1.0))
    form_diff_B = feat_B['recent_form_index'] - feat_A['recent_form_index']
    fc26_diff_B = feat_B['fc26_rating'] - feat_A['fc26_rating']
    qualifiers_ppg_diff_B = feat_B['qualifiers_ppg'] - feat_A['qualifiers_ppg']
    
    lambda_B = poisson_model.predict([[rank_diff_B, log_value_ratio_B, form_diff_B, fc26_diff_B, qualifiers_ppg_diff_B]])[0]
    
    result = (max(0.1, lambda_A), max(0.1, lambda_B))
    lambda_cache[pair] = result
    return result

def predict_match(team_A, team_B, is_knockout=False):
    """
    Predicts a match result by drawing from Poisson goal distributions.
    Returns:
      (goals_A, goals_B, winner)
    """
    lambda_A, lambda_B = get_match_lambda(team_A, team_B)
    
    # Draw goals from Poisson distribution
    goals_A = np.random.poisson(lambda_A)
    goals_B = np.random.poisson(lambda_B)
    
    if goals_A > goals_B:
        return goals_A, goals_B, team_A
    elif goals_B > goals_A:
        return goals_A, goals_B, team_B
    else: # Draw
        if is_knockout:
            # Simulate penalty shootout.
            # Give a slight edge to the team with higher points (overall quality)
            pts_A = team_features[team_A]['fifa_points']
            pts_B = team_features[team_B]['fifa_points']
            p_A_wins = pts_A / (pts_A + pts_B)
            
            winner = np.random.choice([team_A, team_B], p=[p_A_wins, 1.0 - p_A_wins])
            return goals_A, goals_B, winner
        else:
            return goals_A, goals_B, None

def simulate_group_stage(groups):
    """
    Simulates all matches in the group stage.
    Returns:
      advanced_teams (list of 32 teams: top 2 of each of the 12 groups + 8 best third places)
    """
    group_tables = {}
    
    # Initialize group standing tables
    for group_name, teams in groups.items():
        group_tables[group_name] = {
            t: {"points": 0, "goals_scored": 0, "goals_conceded": 0, "goal_diff": 0} 
            for t in teams
        }
        
    # Play matches
    for group_name, teams in groups.items():
        table = group_tables[group_name]
        n_teams = len(teams)
        for i in range(n_teams):
            for j in range(i + 1, n_teams):
                t1, t2 = teams[i], teams[j]
                g1, g2, winner = predict_match(t1, t2, is_knockout=False)
                
                # Update table
                table[t1]["goals_scored"] += g1
                table[t1]["goals_conceded"] += g2
                table[t1]["goal_diff"] = table[t1]["goals_scored"] - table[t1]["goals_conceded"]
                
                table[t2]["goals_scored"] += g2
                table[t2]["goals_conceded"] += g1
                table[t2]["goal_diff"] = table[t2]["goals_scored"] - table[t2]["goals_conceded"]
                
                if winner == t1:
                    table[t1]["points"] += 3
                elif winner == t2:
                    table[t2]["points"] += 3
                else:
                    table[t1]["points"] += 1
                    table[t2]["points"] += 1

    # Extract positions
    top_2_teams = []
    third_place_teams = []
    
    for group_name, table in group_tables.items():
        # Sort team standings
        sorted_teams = sorted(
            table.keys(),
            key=lambda t: (table[t]["points"], table[t]["goal_diff"], table[t]["goals_scored"]),
            reverse=True
        )
        
        top_2_teams.extend(sorted_teams[:2])
        
        # Save 3rd place team metrics
        t3 = sorted_teams[2]
        third_place_teams.append({
            "team": t3,
            "points": table[t3]["points"],
            "goal_diff": table[t3]["goal_diff"],
            "goals_scored": table[t3]["goals_scored"]
        })
        
    # Rank and select 8 best third place teams
    sorted_thirds = sorted(
        third_place_teams,
        key=lambda x: (x["points"], x["goal_diff"], x["goals_scored"]),
        reverse=True
    )
    
    best_8_thirds = [x["team"] for x in sorted_thirds[:8]]
    
    # Merge and build list of 32 teams that advance
    # Layout order is important for bracket construction.
    # We organize advanced list as winners, runners-up, and third-places.
    winners = [top_2_teams[i] for i in range(0, len(top_2_teams), 2)]
    runners = [top_2_teams[i] for i in range(1, len(top_2_teams), 2)]
    
    return winners, runners, best_8_thirds

def simulate_knockout_stage(winners, runners, thirds):
    """
    Runs the knockout bracket: Round of 32 -> Round of 16 -> QF -> SF -> Final.
    Returns:
      stages_reached (dict mapping each team to the furthest stage they reached)
    """
    # 1. Round of 32 pairing (16 matches)
    # T stands for the 8 best 3rd place teams
    T = thirds
    
    # Pairings in a deterministic layout covering all teams
    r32_matches = [
        (winners[0], T[0]),   # W_A vs T_1
        (runners[6], runners[7]), # R_G vs R_H
        (winners[1], T[1]),   # W_B vs T_2
        (runners[8], runners[9]), # R_I vs R_J
        (winners[2], T[2]),   # W_C vs T_3
        (runners[10], runners[11]), # R_K vs R_L
        (winners[3], T[3]),   # W_D vs T_4
        (runners[4], runners[5]), # R_E vs R_F
        (winners[4], T[4]),   # W_E vs T_5
        (winners[7], runners[0]), # W_H vs R_A
        (winners[5], T[6]),   # W_F vs T_7
        (winners[8], runners[1]), # W_I vs R_B
        (winners[6], T[5]),   # W_G vs T_6
        (winners[9], runners[2]), # W_J vs R_C
        (winners[10], T[7]),  # W_K vs T_8
        (winners[11], runners[3]) # W_L vs R_D
    ]
    
    stages = {}
    
    # Round of 32
    r16_teams = []
    for t1, t2 in r32_matches:
        _, _, winner = predict_match(t1, t2, is_knockout=True)
        r16_teams.append(winner)
        loser = t1 if winner == t2 else t2
        stages[loser] = "Round of 32"
        
    # Round of 16 (8 matches)
    qf_teams = []
    for i in range(0, len(r16_teams), 2):
        t1, t2 = r16_teams[i], r16_teams[i+1]
        _, _, winner = predict_match(t1, t2, is_knockout=True)
        qf_teams.append(winner)
        loser = t1 if winner == t2 else t2
        stages[loser] = "Round of 16"
        
    # Quarter-Finals (4 matches)
    sf_teams = []
    for i in range(0, len(qf_teams), 2):
        t1, t2 = qf_teams[i], qf_teams[i+1]
        _, _, winner = predict_match(t1, t2, is_knockout=True)
        sf_teams.append(winner)
        loser = t1 if winner == t2 else t2
        stages[loser] = "Quarter-Finals"
        
    # Semi-Finals (2 matches)
    final_teams = []
    for i in range(0, len(sf_teams), 2):
        t1, t2 = sf_teams[i], sf_teams[i+1]
        _, _, winner = predict_match(t1, t2, is_knockout=True)
        final_teams.append(winner)
        loser = t1 if winner == t2 else t2
        stages[loser] = "Semi-Finals"
        
    # Final
    t1, t2 = final_teams[0], final_teams[1]
    _, _, champion = predict_match(t1, t2, is_knockout=True)
    runner_up = t1 if champion == t2 else t2
    
    stages[runner_up] = "Runner-Up"
    stages[champion] = "Champion"
    
    return stages

def run_monte_carlo(n_simulations=1000):
    """
    Runs the World Cup simulation n times and returns aggregated statistics.
    """
    print(f"=== Running Monte Carlo Simulation ({n_simulations} iterations) ===")
    
    # Load groups configuration
    with open(config.GROUPS_JSON_PATH, 'r') as f:
        groups = json.load(f)
        
    # Initialize trackers
    # Metrics to track: Group stage qualification, R16, QF, SF, Final, Champion
    results = {
        t: {
            "Qualified Group": 0,
            "Round of 16": 0,
            "Quarter-Finals": 0,
            "Semi-Finals": 0,
            "Runner-Up": 0,
            "Champion": 0
        }
        for t in config.QUALIFIED_TEAMS
    }
    
    for sim in range(n_simulations):
        if (sim + 1) % 5000 == 0 or sim == 0 or sim == n_simulations - 1:
            pct = (sim + 1) / n_simulations * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"\rProgress: [{bar}] {pct:.1f}% ({sim + 1}/{n_simulations})", end="", flush=True)
            if sim == n_simulations - 1:
                print()  # Newline at completion
                
        # 1. Simulate Group Stage
        winners, runners, thirds = simulate_group_stage(groups)
        
        # Track group qualifications
        advanced_in_sim = set(winners + runners + thirds)
        for t in advanced_in_sim:
            results[t]["Qualified Group"] += 1
            
        # 2. Simulate Knockout Stage
        knockout_stages = simulate_knockout_stage(winners, runners, thirds)
        
        for team, furthest_stage in knockout_stages.items():
            if furthest_stage == "Champion":
                results[team]["Champion"] += 1
            elif furthest_stage == "Runner-Up":
                results[team]["Runner-Up"] += 1
            elif furthest_stage == "Semi-Finals":
                results[team]["Semi-Finals"] += 1
            elif furthest_stage == "Quarter-Finals":
                results[team]["Quarter-Finals"] += 1
            elif furthest_stage == "Round of 16":
                results[team]["Round of 16"] += 1
                
    # Normalize counts to percentages/probabilities
    for team in results:
        # Sum cumulative counts: team reaching Champion also reached SF, QF, R16, Group
        champion_cnt = results[team]["Champion"]
        runner_up_cnt = results[team]["Runner-Up"]
        sf_cnt = results[team]["Semi-Finals"]
        qf_cnt = results[team]["Quarter-Finals"]
        r16_cnt = results[team]["Round of 16"]
        
        # Cumulative
        results[team]["Finalist_pct"] = (champion_cnt + runner_up_cnt) / n_simulations
        results[team]["SF_pct"] = (champion_cnt + runner_up_cnt + sf_cnt) / n_simulations
        results[team]["QF_pct"] = (champion_cnt + runner_up_cnt + sf_cnt + qf_cnt) / n_simulations
        results[team]["R16_pct"] = (champion_cnt + runner_up_cnt + sf_cnt + qf_cnt + r16_cnt) / n_simulations
        results[team]["Qualified_pct"] = results[team]["Qualified Group"] / n_simulations
        results[team]["Champion_pct"] = champion_cnt / n_simulations
        
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results = df_results.drop(columns=["Qualified Group", "Round of 16", "Quarter-Finals", "Semi-Finals", "Runner-Up", "Champion"])
    df_results = df_results.rename(columns={
        "Qualified_pct": "Qualified Group Stage",
        "R16_pct": "Reached Round of 16",
        "QF_pct": "Reached Quarter-Finals",
        "SF_pct": "Reached Semi-Finals",
        "Finalist_pct": "Reached Final",
        "Champion_pct": "Winner"
    })
    
    return df_results

if __name__ == "__main__":
    df_res = run_monte_carlo(100)
    print("\n=== TOP 10 SIMULATION RESULTS ===")
    print(df_res.sort_values(by="Winner", ascending=False).head(10))
