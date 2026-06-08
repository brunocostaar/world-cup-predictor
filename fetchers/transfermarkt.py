import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import sys

# Add parent dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def parse_value_to_millions(val_str):
    """
    Converts Transfermarkt squad value strings (e.g. '€1.55bn', '€982.00m')
    into numerical float values in Millions of Euros.
    """
    if not val_str or pd.isna(val_str):
        return 0.0
    val_str = str(val_str).strip().lower()
    if val_str == '-' or val_str == '':
        return 0.0
        
    # Remove currency and whitespace
    val_str = val_str.replace('€', '').replace('$', '').replace('£', '').strip()
    
    # Determine multiplier
    multiplier = 1.0
    if 'bn' in val_str or 'mrd' in val_str or 'billion' in val_str:
        multiplier = 1000.0
        val_str = val_str.replace('bn', '').replace('mrd', '').replace('billion', '').strip()
    elif 'm' in val_str or 'mio' in val_str or 'million' in val_str:
        multiplier = 1.0
        val_str = val_str.replace('m', '').replace('mio', '').replace('million', '').strip()
    elif 'k' in val_str or 'thousand' in val_str:
        multiplier = 0.001
        val_str = val_str.replace('k', '').replace('thousand', '').strip()

    # Clean separators
    # Case 1: Both dot and comma (e.g., "1.550,50" or "1,550.50")
    if '.' in val_str and ',' in val_str:
        dot_idx = val_str.find('.')
        comma_idx = val_str.find(',')
        if dot_idx < comma_idx: # Dot is thousands, comma is decimal
            val_str = val_str.replace('.', '').replace(',', '.')
        else: # Comma is thousands, dot is decimal
            val_str = val_str.replace(',', '')
    # Case 2: Only comma (e.g., "1,55" or "1,550")
    elif ',' in val_str:
        parts = val_str.split(',')
        if len(parts[-1]) == 3 and len(parts) > 1 and len(parts[0]) <= 3: # thousands separator (e.g. "1,550")
            val_str = val_str.replace(',', '')
        else:
            val_str = val_str.replace(',', '.')
            
    try:
        return float(val_str) * multiplier
    except ValueError:
        return 0.0


def fetch_squad_values(force_fallback=False):
    """
    Fetches the market values of national teams from Transfermarkt.
    Scrapes the statistics pages and falls back to seeded CSV on block/failure.
    """
    print("=== Fetching Squad Values ===")
    
    if force_fallback:
        print("Forced fallback enabled. Loading seeded squad values...")
        return load_seed_squad_values()

    scraped_data = []
    
    try:
        # Transfermarkt splits the list of valuable national teams into pages (typically 25 teams per page)
        # We fetch the first 2 pages (50 teams total) to cover the 48 World Cup participants.
        for page in [1, 2]:
            url = f"{config.TRANSFERMARKT_URL}?page={page}"
            print(f"Requesting Transfermarkt page {page} from {url}...")
            
            # Add a small polite sleep to avoid hitting rate limits too quickly
            if page > 1:
                time.sleep(1.5)
                
            response = requests.get(url, headers=config.SCRAPING_HEADERS, timeout=10)
            
            if response.status_code == 403:
                print("Access forbidden (403). Transfermarkt is blocking the scraper. Using seed data...")
                return load_seed_squad_values()
            elif response.status_code != 200:
                raise Exception(f"Failed to fetch page {page}. Status: {response.status_code}")
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the main data table
            table = soup.find('table', class_='items')
            if not table:
                table = soup.find('div', class_='responsive-table')
                if table:
                    table = table.find('table')
            
            if not table:
                print(f"Could not find table on Transfermarkt page {page}. Using seed...")
                return load_seed_squad_values()
                
            rows = table.find('tbody').find_all('tr') if table.find('tbody') else table.find_all('tr')[1:]
            
            for row in rows:
                # Filter out separator or spacer rows
                if 'class' in row.attrs and any(c in ['odd', 'even'] for c in row.attrs['class']):
                    cols = row.find_all('td')
                    if len(cols) >= 3:
                        # Team name is typically inside a 'hauptlink' cell containing a link
                        team_cell = row.find('td', class_='hauptlink')
                        if team_cell:
                            team_name = team_cell.text.strip()
                            # Market value is in a 'rechts hauptlink' cell
                            value_cell = row.find('td', class_='rechts hauptlink')
                            if value_cell:
                                value_str = value_cell.text.strip()
                                value_m = parse_value_to_millions(value_str)
                                scraped_data.append({
                                    "team": team_name,
                                    "squad_value_m": value_m
                                })
                                
        if len(scraped_data) > 0:
            df = pd.DataFrame(scraped_data)
            # Remove duplicate rows if any
            df = df.drop_duplicates(subset=['team'])
            df.to_csv(config.SQUAD_OUTPUT_PATH, index=False)
            print(f"Successfully scraped {len(df)} squad values from Transfermarkt.")
            return df
        else:
            print("No teams scraped from Transfermarkt page structure. Using seed data...")
            return load_seed_squad_values()

    except Exception as e:
        print(f"Scraping Transfermarkt failed due to: {e}. Falling back to seeded values.")
        return load_seed_squad_values()

def load_seed_squad_values():
    """Loads seeded squad values dataset."""
    if not os.path.exists(config.SQUAD_SEED_PATH):
        raise FileNotFoundError(f"Squad seed file not found at {config.SQUAD_SEED_PATH}")
        
    df = pd.read_csv(config.SQUAD_SEED_PATH)
    df.to_csv(config.SQUAD_OUTPUT_PATH, index=False)
    print(f"Seeded squad values saved to {config.SQUAD_OUTPUT_PATH} (Total: {len(df)} teams).")
    return df

if __name__ == "__main__":
    fetch_squad_values()
