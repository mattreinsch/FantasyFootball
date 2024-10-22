import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_model():
    # Load engineered features
    df = pd.read_csv('data/features.csv')
    X = df[['points_per_game', 'touchdown_rate', 'yard_per_attempt']]
    y = df['fantasy_points']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost model
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    # Save the trained model
    model.save_model('data/fantasy_model.json')
    print("Model training completed.")
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model Mean Squared Error: {mse}")
    
    return model

if __name__ == "__main__":
    train_model()