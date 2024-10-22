import pandas as pd
import xgboost as xgb

def make_predictions(input_data):
    # Load the trained model
    model = xgb.XGBRegressor()
    model.load_model('data/fantasy_model.json')
    
    # Make predictions
    predictions = model.predict(input_data)
    return predictions

if __name__ == "__main__":
    # Example input data
    input_data = pd.DataFrame({
        'points_per_game': [15.2, 18.5],
        'touchdown_rate': [0.1, 0.2],
        'yard_per_attempt': [5.0, 6.5]
    })
    predictions = make_predictions(input_data)
    print(predictions)