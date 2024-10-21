import pandas as pd
import requests

def fetch_player_stats():
    # Example function for collecting player stats from a CSV or API
    url = 'player-stats.csv'
    df = pd.read_csv(url)
    df.to_csv('data/player_stats.csv', index=False)
    print("Player stats downloaded successfully.")
    return df

def fetch_injury_reports():
    # Example function for fetching injury reports
    url = 'injury-reports.csv'
    df = pd.read_csv(url)
    df.to_csv('data/injury_reports.csv', index=False)
    print("Injury reports downloaded successfully.")
    return df

if __name__ == "__main__":
    fetch_player_stats()
    fetch_injury_reports()