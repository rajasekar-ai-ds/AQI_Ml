import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib

daily = pd.read_csv("finalized_data.csv", parse_dates=['Date'])

train = daily[daily['Date'] < '2019-07-01']
test = daily[daily['Date'] >= '2019-07-01']

features = [c for c in daily.columns if c not in ('Date', 'AQI_next')]
X_train, y_train = train[features], train['AQI_next']
X_test, y_test = test[features], test['AQI_next']


base_pred = X_test['AQI']
print('Baseline MAE :', mean_absolute_error(y_test, base_pred))
print('Baseline RMSE:', root_mean_squared_error(y_test, base_pred))

rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, rf_pred)}")
print(f"RMSE: {root_mean_squared_error(y_test, rf_pred)}")

joblib.dump({'model': rf, 'features': features}, 'aqi_rf.joblib', compress=3)