import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    # Load data
    player_stats = pd.read_csv('data/player_stats.csv')
    injury_reports = pd.read_csv('data/injury_reports.csv')
    
    # preprocessing steps
    player_stats.fillna(0, inplace=True)
    player_stats['injury_status'] = player_stats['player_name'].apply(
        lambda x: 'Out' if x in injury_reports['player_name'].values else 'Active'
    )
    
    # Normalize numerical columns
    numeric_cols = ['passing_yards', 'rushing_yards', 'receiving_yards']
    scaler = StandardScaler()
    player_stats[numeric_cols] = scaler.fit_transform(player_stats[numeric_cols])
    
    player_stats.to_csv('data/processed_player_stats.csv', index=False)
    print("Data preprocessing completed.")
    return player_stats

if __name__ == "__main__":
    preprocess_data()