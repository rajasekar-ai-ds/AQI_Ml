from datetime import date
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from datetime import timedelta

app = FastAPI()

bundle = joblib.load('aqi_rf.joblib')
model, features = bundle['model'], bundle['features']
df = pd.read_csv('finalized_data.csv', parse_dates=['Date'])

CITIES = {'Bengaluru', 'Chennai', 'Delhi', 'Hyderabad', 'Visakhapatnam'}
TEST_START = pd.Timestamp('2019-07-01')

@app.get('/predict')
def predict(city: str, day: date):
    city = city.strip().title()
    if city not in CITIES:
        raise HTTPException(404, "The city you entered is not supported by our application")
    if (pd.Timestamp(day)) < TEST_START:
        raise HTTPException(422, detail=f"We only provide service starting from {TEST_START.date()}")

    row = df[(df[f'city_{city}'] == 1) & (df['Date'] == pd.Timestamp(day))]

    if row.empty:
        raise HTTPException(404, "No data for that day")

    pred = round(float(model.predict(row[features])[0]), 1)
    actual = float(row['AQI_next'].iloc[0])
    baseline = float(row['AQI'].iloc[0])
    return {'city': city, 'date': day, 'predicted': pred, 'actual': actual, 'baseline': baseline,
            'forecast_for': day + timedelta(days=1),'model_error': round(abs(pred - actual), 1),
            'baseline_error': round(abs(baseline - actual), 1),}

