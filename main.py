from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import datetime
import pandas as pd
import requests
import os
from dotenv import load_dotenv
import logging
import joblib
from tensorflow.keras.models import load_model

load_dotenv()  # Load environment variables from .env file

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load preprocessing objects and model
pca = joblib.load('pca.pkl')
power_transformer = joblib.load('power_transformer.pkl')
scaler = joblib.load('scaler.pkl')
model = load_model('best_pedestrian_prediction_model.h5')

def fetch_data(hours=1):
    now = datetime.datetime.now(datetime.timezone.utc)
    end_time = now.replace(minute=0, second=0, microsecond=0) - datetime.timedelta(hours=1)
    start_time = end_time - datetime.timedelta(hours=hours)

    start_time_str = start_time.isoformat()
    end_time_str = end_time.isoformat()

    url = "https://api.hystreet.com/locations/257"
    querystring = {"from": start_time_str, "to": end_time_str, "resolution": "hour"}
    headers = {
        "Content-Type": "application/json",
        "X-API-Token": os.getenv("API_TOKEN")
    }

    logger.info(f"Fetching data from {url} with params {querystring} and headers {headers}")

    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        data = response.json()
        if 'measurements' not in data:
            raise ValueError("The key 'measurements' is not in the JSON response.")
        return data
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Request failed: {e}")
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def convert_to_dataframe(data):
    dataset_columns = [
        'location', 'timestamp', 'weekday', 'pedestrians_count', 'weather_condition', 'temperature'
    ]

    rows = []
    for measurement in data['measurements']:
        timestamp = measurement['timestamp']
        row = [
            '257',  # location ID as provided in the URL
            timestamp,  # timestamp
            datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%f%z').strftime('%A'),  # weekday
            measurement['pedestrians_count'],  # pedestrians count
            measurement['weather_condition'],  # weather condition
            measurement['temperature']  # temperature
        ]
        rows.append(row)

    df = pd.DataFrame(rows, columns=dataset_columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%f%z')
    return df

def preprocess_data(df):
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['is_holiday'] = df['timestamp'].dt.date.apply(lambda x: x in holidays.Germany())
    df['weekday_code'] = df['timestamp'].dt.weekday

    season_map = {12: 'winter', 1: 'winter', 2: 'winter',
                  3: 'spring', 4: 'spring', 5: 'spring',
                  6: 'summer', 7: 'summer', 8: 'summer',
                  9: 'autumn', 10: 'autumn', 11: 'autumn'}
    df['season'] = df['month'].map(season_map)

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['weather_condition', 'season'], drop_first=True)
    
    # Ensure all expected columns are present
    expected_columns = [
        'temperature', 'hour', 'day', 'month', 'is_holiday', 'weekday_code',
        'weather_condition_clear-day', 'weather_condition_clear-night', 'weather_condition_cloudy',
        'weather_condition_fog', 'weather_condition_partly-cloudy-day', 'weather_condition_partly-cloudy-night',
        'weather_condition_rain', 'weather_condition_snow', 'weather_condition_wind',
        'season_spring', 'season_summer', 'season_winter'
    ]
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Standardize features
    features = df[expected_columns].values
    features_scaled = scaler.transform(features)
    
    # Apply power transformation
    features_power_transformed = power_transformer.transform(features_scaled)
    
    # Apply PCA
    features_pca = pca.transform(features_power_transformed)
    
    return features_pca

@app.get("/fetch_last_hour_data")
def fetch_last_hour_data():
    try:
        raw_data = fetch_data(hours=1)
        df = convert_to_dataframe(raw_data)
        return df.to_dict()
    except HTTPException as e:
        logger.error(f"HTTPException occurred: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error occurred: {e}")

@app.post("/predict")
def predict(data: dict):
    try:
        df = convert_to_dataframe({'measurements': [data]})
        preprocessed_data = preprocess_data(df)
        prediction = model.predict(preprocessed_data)
        return {"prediction": prediction[0][0]}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
