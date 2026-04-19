import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="Smart Traffic Advisory System", layout="wide")

st.markdown("""
<style>
body { font-family: Arial, sans-serif; }
.card { padding: 22px; border-radius: 16px; background: #f7f9fc; border: 1px solid #e5e7eb; }
.big-number { font-size: 42px; font-weight: 700; margin: 0; }
.label { color: #6b7280; font-size: 14px; }
.status { font-size: 22px; font-weight: 700; margin-top: 10px; }
.low { color: #0f766e; }
.medium { color: #b45309; }
.high { color: #b91c1c; }
.advice { padding: 14px 18px; border-radius: 12px; margin-top: 12px; font-weight: 600; }
.advice-low { background: #ecfdf5; color: #065f46; }
.advice-medium { background: #fffbeb; color: #92400e; }
.advice-high { background: #fef2f2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = joblib.load("traffic_level_model.pkl")
    columns = joblib.load("traffic_model_columns.pkl")
    location_cols = joblib.load("location_columns.pkl")
    return model, columns, location_cols

model, train_columns, location_cols = load_assets()
location_options = [c.replace("location_id_", "") for c in location_cols]

def get_advice(level):
    if level == "Low":
        return "Traffic is smooth. Travel is suitable right now.", "advice-low"
    elif level == "Medium":
        return "Traffic may slow down. Plan a little extra time.", "advice-medium"
    return "Heavy congestion likely. Delay travel or choose an alternate route.", "advice-high"

def risk_score(level):
    return {"Low": 25, "Medium": 60, "High": 90}[level]

st.title("Smart Traffic Advisory System")
st.write("Select place, date, time, weather, signal status, and accident information to predict traffic level.")

col1, col2 = st.columns([1, 1])

with col1:
    selected_location = st.selectbox("Select place / location id", location_options)
    selected_date = st.date_input("Select date")
    selected_time = st.time_input("Select time")
    weather_condition = st.selectbox("Weather condition", ["Clear", "Cloudy", "Rain", "Fog", "Storm"])
    signal_status = st.selectbox("Signal status", ["Normal", "Slow", "Malfunction"])
    accident_reported = st.selectbox("Accident reported", ["No", "Yes"])
    predict_btn = st.button("Predict")

with col2:
    if predict_btn:
        dt = datetime.combine(selected_date, selected_time)
        hour = dt.hour
        day = dt.day
        month = dt.month
        weekday = dt.weekday()
        is_weekend = 1 if weekday >= 5 else 0
        is_rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18, 19] else 0
        is_night = 1 if hour in [0, 1, 2, 3, 4, 5] else 0

        row = pd.DataFrame([{
            "hour": hour,
            "day": day,
            "month": month,
            "weekday": weekday,
            "is_weekend": is_weekend,
            "is_rush_hour": is_rush_hour,
            "is_night": is_night,
            "location_id": selected_location,
            "weather_condition": weather_condition,
            "signal_status": signal_status,
            "accident_reported": accident_reported,
            "temperature": 25,
            "humidity": 50,
            "avg_vehicle_speed": 30,
            "vehicle_count_cars": 100,
            "vehicle_count_trucks": 20,
            "vehicle_count_bikes": 10
        }])

        data = pd.get_dummies(row).reindex(columns=train_columns, fill_value=0)
        pred_class = int(model.predict(data)[0])

        if pred_class == 0:
            level = "Low"
            cls = "low"
        elif pred_class == 1:
            level = "Medium"
            cls = "medium"
        else:
            level = "High"
            cls = "high"

        score = risk_score(level)
        advice_text, advice_cls = get_advice(level)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<p class="label">Predicted traffic level</p><p class="big-number">{level}</p><div class="status {cls}">Class: {pred_class}</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="label">Risk Score</p><p class="big-number">{score}</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="advice {advice_cls}">{advice_text}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.write(f"Selected Location: {selected_location}")
        st.write(f"Weather: {weather_condition}")
        st.write(f"Signal Status: {signal_status}")
        st.write(f"Accident Reported: {accident_reported}")

        hours = list(range(24))
        demo_preds = []
        for h in hours:
            temp = pd.DataFrame([{
                "hour": h,
                "day": day,
                "month": month,
                "weekday": weekday,
                "is_weekend": is_weekend,
                "is_rush_hour": 1 if h in [7, 8, 9, 16, 17, 18, 19] else 0,
                "is_night": 1 if h in [0, 1, 2, 3, 4, 5] else 0,
                "location_id": selected_location,
                "weather_condition": weather_condition,
                "signal_status": signal_status,
                "accident_reported": accident_reported,
                "temperature": 25,
                "humidity": 50,
                "avg_vehicle_speed": 30,
                "vehicle_count_cars": 100,
                "vehicle_count_trucks": 20,
                "vehicle_count_bikes": 10
            }])
            temp = pd.get_dummies(temp).reindex(columns=train_columns, fill_value=0)
            demo_preds.append(int(model.predict(temp)[0]))

        fig = px.line(x=hours, y=demo_preds, markers=True, title="Traffic Level Pattern by Hour")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})