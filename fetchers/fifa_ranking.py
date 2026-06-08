import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import sys

# Add parent dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def fetch_fifa_rankings(force_fallback=False):
    """
    Fetches the latest FIFA Men's Rankings.
    Tries to scrape or load from the web, and falls back to a high-quality seed file on failure.
    """
    print("=== Fetching FIFA Rankings ===")
    
    if force_fallback:
        print("Forced fallback enabled. Loading seeded FIFA rankings...")
        return load_seed_rankings()

    try:
        # Note: FIFA rankings are dynamically loaded via JavaScript/APIs.
        # We try to request the page and parse any static table elements if available,
        # but have a robust fallback mechanism since contentapi changes frequently.
        print(f"Requesting FIFA Rankings from {config.FIFA_RANKING_URL}...")
        response = requests.get(config.FIFA_RANKING_URL, headers=config.SCRAPING_HEADERS, timeout=10)
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch FIFA page. Status code: {response.status_code}")
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for typical table structures or ranking list items
        # Usually, FIFA ranking tables have tables with classes containing 'table' or rows 'tbl-row'
        rows = []
        table = soup.find('table')
        if table:
            for tr in table.find_all('tr')[1:]: # Skip header
                cols = tr.find_all('td')
                if len(cols) >= 3:
                    rank = cols[0].text.strip()
                    # Try to find team name inside a link or text
                    team_el = cols[1].find('span', class_='tbl-teamname__name') or cols[1]
                    team = team_el.text.strip()
                    points_el = cols[2].find('span') or cols[2]
                    points = points_el.text.strip()
                    rows.append({"team": team, "rank": rank, "points": points})

        if len(rows) > 0:
            df = pd.DataFrame(rows)
            # Basic cleaning
            df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
            df['points'] = pd.to_numeric(df['points'], errors='coerce')
            df = df.dropna(subset=['team', 'rank'])
            
            # Save scraped data
            df.to_csv(config.FIFA_OUTPUT_PATH, index=False)
            print(f"Successfully scraped {len(df)} teams from FIFA website.")
            return df
        else:
            print("FIFA page did not return standard static rows (dynamically loaded). Using seed rankings...")
            return load_seed_rankings()

    except Exception as e:
        print(f"Scraping failed due to: {e}. Falling back to seeded FIFA rankings.")
        return load_seed_rankings()

def load_seed_rankings():
    """Loads seeded FIFA ranking dataset."""
    if not os.path.exists(config.FIFA_SEED_PATH):
        raise FileNotFoundError(f"FIFA seed file not found at {config.FIFA_SEED_PATH}")
        
    df = pd.read_csv(config.FIFA_SEED_PATH)
    df.to_csv(config.FIFA_OUTPUT_PATH, index=False)
    print(f"Seeded FIFA rankings saved to {config.FIFA_OUTPUT_PATH} (Total: {len(df)} teams).")
    return df

if __name__ == "__main__":
    fetch_fifa_rankings()
