from datetime import date

import requests
import streamlit as st

API_URL = "https://aqi-ml-ymp9.onrender.com/predict"

CITIES = ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Visakhapatnam"]
MIN_DATE = date(2019, 7, 1)
MAX_DATE = date(2020, 6, 30)  # last date in the test data

st.title("Next-Day AQI Forecast")

city = st.selectbox("City", CITIES)
day = st.date_input("Date", value=MIN_DATE, min_value=MIN_DATE, max_value=MAX_DATE)

if st.button("Predict"):
    try:
        with st.spinner("Waking up the model server (free tier, may take ~1 min)..."):
            resp = requests.get(API_URL, params={"city": city, "day": day.isoformat()}, timeout=90)
    except requests.exceptions.Timeout:
        st.error("The server took too long to respond. It may still be waking up - please try again in a moment.")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the model server. Please check your connection and try again.")
    else:
        if resp.status_code == 200:
            data = resp.json()
            st.subheader(f"AQI forecast for {data['forecast_for']}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted AQI", data["predicted"])
            col2.metric("Actual AQI", data["actual"])
            col3.metric("Baseline AQI", data["baseline"])

            model_err, base_err = data["model_error"], data["baseline_error"]
            col4, col5 = st.columns(2)
            col4.metric("Model error", model_err)
            col5.metric("Baseline error", base_err)

            if model_err < base_err:
                st.success(f"The model was closer to the actual AQI (by {base_err - model_err:.1f}).")
            elif base_err < model_err:
                st.info(f"The baseline was closer to the actual AQI (by {model_err - base_err:.1f}).")
            else:
                st.info("The model and baseline were equally close.")
        elif resp.status_code in (404, 422):
            detail = resp.json().get("detail", resp.text)
            st.warning(detail if isinstance(detail, str) else str(detail))
        else:
            st.error(f"Unexpected response from server (HTTP {resp.status_code}). Please try again.")