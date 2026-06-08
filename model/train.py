import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PoissonRegressor
import pickle
import sys

# Add parent dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_training_data():
    """Loads historical matches and engineers features for ML training."""
    from processing.prepare_dataset import normalize_team_name
    
    csv_path = os.path.join(config.DATA_DIR, "historical_matches.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training dataset not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Load Metadata (FC 26 Rating and Qualifiers PPG)
    meta_df = pd.read_csv(config.METADATA_OUTPUT_PATH)
    meta_lookup = meta_df.set_index('team').to_dict(orient='index')
    
    def get_meta(team_name, rank):
        clean_name = normalize_team_name(team_name)
        if clean_name in meta_lookup:
            return meta_lookup[clean_name]['fc26_rating'], meta_lookup[clean_name]['qualifiers_ppg']
        else:
            est_rating = int(max(60, min(87, 87 - 0.22 * rank)))
            est_ppg = round(max(0.8, min(2.8, 2.6 - 0.018 * rank)), 2)
            return est_rating, est_ppg
            
    # Apply metadata mapping
    home_meta = [get_meta(row['home_team'], row['home_rank']) for idx, row in df.iterrows()]
    away_meta = [get_meta(row['away_team'], row['away_rank']) for idx, row in df.iterrows()]
    
    df['home_fc26_rating'] = [m[0] for m in home_meta]
    df['home_qualifiers_ppg'] = [m[1] for m in home_meta]
    df['away_fc26_rating'] = [m[0] for m in away_meta]
    df['away_qualifiers_ppg'] = [m[1] for m in away_meta]
    
    # Feature Engineering
    # 1. Rank difference: away_rank - home_rank.
    # Positive means Home team is better ranked (lower rank index).
    df['rank_diff'] = df['away_rank'] - df['home_rank']
    
    # 2. Squad value ratio: home_value / away_value.
    df['value_ratio'] = df['home_value_m'] / (df['away_value_m'] + 1.0)
    df['log_value_ratio'] = np.log10(df['value_ratio'])
    
    # 3. Recent form difference: home_form - away_form.
    df['form_diff'] = df['home_form'] - df['away_form']
    
    # 4. FC 26 Rating difference and Qualifiers PPG difference
    df['fc26_diff'] = df['home_fc26_rating'] - df['away_fc26_rating']
    df['qualifiers_ppg_diff'] = df['home_qualifiers_ppg'] - df['away_qualifiers_ppg']
    
    # Classification target: 1 (Home Win), 0 (Draw), -1 (Away Win)
    df['result'] = np.where(df['home_goals'] > df['away_goals'], 1,
                            np.where(df['home_goals'] == df['away_goals'], 0, -1))
                            
    return df

def train_models():
    """Trains a Random Forest Classifier and Poisson Goal Regressors."""
    print("=== Training Machine Learning Models ===")
    
    # Load and prepare data
    df = load_training_data()
    
    # Define features and targets (Added fc26_diff and qualifiers_ppg_diff)
    features = ['rank_diff', 'log_value_ratio', 'form_diff', 'fc26_diff', 'qualifiers_ppg_diff']
    X = df[features]
    y_class = df['result']
    y_home_goals = df['home_goals']
    y_away_goals = df['away_goals']
    
    # 1. Train Random Forest Classifier
    # RF works by building multiple decision trees that vote on the final outcome.
    # We set random_state for reproducibility.
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X.values, y_class)
    
    # Calculate training accuracy
    class_acc = rf_model.score(X.values, y_class)
    print(f"Random Forest Classifier Accuracy on training data: {class_acc * 100:.1f}%")
    
    # Feature Importance Explanation
    importances = rf_model.feature_importances_
    print("\n--- Feature Importance (Random Forest) ---")
    for feature, importance in zip(features, importances):
        print(f"Feature '{feature:<15}': {importance * 100:.1f}% importance")
    print("------------------------------------------")
    
    # 2. Train Symmetric Poisson Regressor for Goals
    # Poisson regression is ideal for modeling count data (goals).
    # We construct a symmetric dataset with two rows per match (one from each team's perspective)
    # to avoid home-away asymmetries on neutral ground. We use alpha=0.1 to regularize
    # volatile features like recent form.
    print("Preparing symmetric dataset for Poisson model training...")
    sym_rows = []
    for _, row in df.iterrows():
        # Home team perspective
        sym_rows.append({
            'goals': row['home_goals'],
            'rank_diff': row['away_rank'] - row['home_rank'],
            'log_value_ratio': np.log10(row['home_value_m'] / (row['away_value_m'] + 1.0)),
            'form_diff': row['home_form'] - row['away_form'],
            'fc26_diff': row['home_fc26_rating'] - row['away_fc26_rating'],
            'qualifiers_ppg_diff': row['home_qualifiers_ppg'] - row['away_qualifiers_ppg']
        })
        # Away team perspective
        sym_rows.append({
            'goals': row['away_goals'],
            'rank_diff': row['home_rank'] - row['away_rank'],
            'log_value_ratio': np.log10(row['away_value_m'] / (row['home_value_m'] + 1.0)),
            'form_diff': row['away_form'] - row['home_form'],
            'fc26_diff': row['away_fc26_rating'] - row['home_fc26_rating'],
            'qualifiers_ppg_diff': row['away_qualifiers_ppg'] - row['home_qualifiers_ppg']
        })
    sym_df = pd.DataFrame(sym_rows)
    X_sym = sym_df[features].values
    y_goals = sym_df['goals'].values
    
    poisson_model = PoissonRegressor(alpha=0.1)
    poisson_model.fit(X_sym, y_goals)
    
    print("\n--- Symmetric Poisson Model Coefficients ---")
    print(f"Intercept: {poisson_model.intercept_:.4f}")
    for feature, coef in zip(features, poisson_model.coef_):
        print(f"  Feature '{feature:<15}': {coef:.4f} coefficient")
    print("------------------------------------------")
    
    # Save the models using pickle
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    rf_path = os.path.join(config.DATA_DIR, "rf_model.pkl")
    poisson_path = os.path.join(config.DATA_DIR, "poisson_model.pkl")
    
    with open(rf_path, 'wb') as f:
        pickle.dump(rf_model, f)
    with open(poisson_path, 'wb') as f:
        pickle.dump(poisson_model, f)
        
    print(f"\nModels successfully saved to {config.DATA_DIR}:")
    print(f" - Random Forest Classifier: rf_model.pkl")
    print(f" - Symmetric Poisson Regressor: poisson_model.pkl")
    
    return rf_model, poisson_model

if __name__ == "__main__":
    train_models()
