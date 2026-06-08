import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Data File Paths
GROUPS_JSON_PATH = os.path.join(DATA_DIR, "world_cup_2026_groups.json")
FIFA_SEED_PATH = os.path.join(DATA_DIR, "fifa_rankings_seed.csv")
SQUAD_SEED_PATH = os.path.join(DATA_DIR, "squad_values_seed.csv")
METADATA_SEED_PATH = os.path.join(DATA_DIR, "team_metadata_seed.csv")

FIFA_OUTPUT_PATH = os.path.join(DATA_DIR, "fifa_rankings.csv")
SQUAD_OUTPUT_PATH = os.path.join(DATA_DIR, "squad_values.csv")
METADATA_OUTPUT_PATH = os.path.join(DATA_DIR, "team_metadata.csv")
FORM_OUTPUT_PATH = os.path.join(DATA_DIR, "recent_form.csv")
FINAL_DATASET_PATH = os.path.join(DATA_DIR, "final_features.csv")

# Scraping & API Configurations
FIFA_RANKING_URL = "https://www.fifa.com/fifa-world-ranking/ranking-table/men/"
TRANSFERMARKT_URL = "https://www.transfermarkt.com/spieler-statistik/wertvollstenationalmannschaften/plus/0/galerie/0"

# Headers to bypass web blocks (Transfermarkt especially)
SCRAPING_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive"
}

# 48 Qualified Teams Standard Names
QUALIFIED_TEAMS = [
    "Mexico", "South Africa", "South Korea", "Czechia",
    "Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland",
    "Brazil", "Morocco", "Haiti", "Scotland",
    "United States", "Paraguay", "Australia", "Türkiye",
    "Germany", "Curaçao", "Côte d'Ivoire", "Ecuador",
    "Netherlands", "Japan", "Sweden", "Tunisia",
    "Belgium", "Egypt", "Iran", "New Zealand",
    "Spain", "Cabo Verde", "Saudi Arabia", "Uruguay",
    "France", "Senegal", "Iraq", "Norway",
    "Argentina", "Algeria", "Austria", "Jordan",
    "Portugal", "DR Congo", "Uzbekistan", "Colombia",
    "England", "Croatia", "Ghana", "Panama"
]

# Mapping dictionary to unify country names across different data sources
# Key: Standard name, Value: lists of alternative names found in FIFA, Transfermarkt, API-Football
TEAM_NAME_MAPPINGS = {
    "South Korea": ["Korea Republic", "Korea, South", "South Korea", "Rep. of Korea", "Coréia do Sul"],
    "Czechia": ["Czech Republic", "Czechia", "República Checa"],
    "Bosnia and Herzegovina": ["Bosnia-Herzegovina", "Bosnia & Herzegovina", "Bosnia and Herzegovina", "Bósnia"],
    "United States": ["US", "USA", "United States", "United States of America", "Estados Unidos"],
    "Türkiye": ["Turkey", "Türkiye", "Turquia"],
    "Curaçao": ["Curacao", "Curaçao"],
    "Côte d'Ivoire": ["Cote d'Ivoire", "Ivory Coast", "Côte d'Ivoire", "Costa do Marfim"],
    "Iran": ["IR Iran", "Iran", "Iran, Islamic Republic of", "Irã"],
    "Cabo Verde": ["Cape Verde", "Cabo Verde", "Cabo Verde Islands"],
    "DR Congo": ["Congo DR", "Democratic Republic of the Congo", "DR Congo", "RD Congo"],
    "Saudi Arabia": ["Saudi Arabia", "Arabia Saudita"]
}

# Confederation Mapping for 48 Qualified Teams
CONFEDERATION_MAPPING = {
    # UEFA
    "Austria": "UEFA", "Belgium": "UEFA", "Bosnia and Herzegovina": "UEFA", "Croatia": "UEFA",
    "Czechia": "UEFA", "England": "UEFA", "France": "UEFA", "Germany": "UEFA",
    "Netherlands": "UEFA", "Norway": "UEFA", "Portugal": "UEFA", "Scotland": "UEFA",
    "Spain": "UEFA", "Sweden": "UEFA", "Switzerland": "UEFA", "Türkiye": "UEFA",
    # CONMEBOL
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Colombia": "CONMEBOL",
    "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL", "Uruguay": "CONMEBOL",
    # CAF
    "Algeria": "CAF", "Cabo Verde": "CAF", "DR Congo": "CAF", "Côte d'Ivoire": "CAF",
    "Egypt": "CAF", "Ghana": "CAF", "Morocco": "CAF", "Senegal": "CAF",
    "South Africa": "CAF", "Tunisia": "CAF",
    # Concacaf
    "Canada": "Concacaf", "Curaçao": "Concacaf", "Haiti": "Concacaf",
    "Mexico": "Concacaf", "Panama": "Concacaf", "United States": "Concacaf",
    # AFC
    "Australia": "AFC", "Iran": "AFC", "Iraq": "AFC", "Japan": "AFC",
    "Jordan": "AFC", "South Korea": "AFC", "Qatar": "AFC", "Uzbekistan": "AFC",
    # OFC
    "New Zealand": "OFC"
}

# Confederation strength weight factors to adjust for strength of schedule / regional inflation
CONFEDERATION_MULTIPLIERS = {
    "UEFA": 1.0,
    "CONMEBOL": 0.96,
    "CAF": 0.82,
    "Concacaf": 0.80,
    "AFC": 0.76,
    "OFC": 0.65
}

# Squad value adjustments (e.g. discount England homegrown market value premium)
TEAM_VALUE_MULTIPLIERS = {
    "England": 0.75 # 25% discount to account for homegrown inflation
}

