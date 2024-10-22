import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def evaluate_model():
    # Load the trained model
    model = xgb.XGBRegressor()
    model.load_model('data/fantasy_model.json')
    
    # Load test data
    df = pd.read_csv('data/features.csv')
    X_test = df[['points_per_game', 'touchdown_rate', 'yard_per_attempt']]
    y_test = df['fantasy_points']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model Mean Squared Error: {mse}")
    
    return mse

if __name__ == "__main__":
    evaluate_model()