import pandas as pd

def create_features():
    df = pd.read_csv('data/processed_player_stats.csv')
    
    # feature engineering
    df['points_per_game'] = df['fantasy_points'] / df['games_played']
    df['touchdown_rate'] = df['touchdowns'] / df['receptions']
    df['yard_per_attempt'] = df['total_yards'] / df['attempts']
    
    df.to_csv('data/features.csv', index=False)
    print("Feature engineering completed.")
    return df

if __name__ == "__main__":
    create_features()