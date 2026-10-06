import os
from pathlib import Path
from datetime import date

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Shipment Delay AI",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "smart_logistics_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "delay_predictor.pkl"
MODEL_INFO_PATH = PROJECT_ROOT / "models" / "model_info.pkl"

# ============================================================
# CSS
# ============================================================
st.markdown(
    """
    <style>
    .stApp { background-color: #F8FAFC; }
    section[data-testid="stSidebar"] { background-color: #0F172A; }

    /* Responsive layout: desktop -> tablet -> mobile */
    .block-container { width:100%; max-width:1600px; padding-top:2rem; padding-left:clamp(1rem,2.5vw,3rem); padding-right:clamp(1rem,2.5vw,3rem); }
    .main-title { font-size:clamp(24px,3vw,36px); line-height:1.15; }
    .subtitle { font-size:clamp(13px,1.5vw,16px); }
    .section-title { font-size:clamp(18px,2vw,22px); }
    .kpi-value, .ai-value { font-size:clamp(22px,2.5vw,30px); overflow-wrap:anywhere; }
    .stButton > button, .stFormSubmitButton > button { min-height:44px; width:100%; white-space:normal; }
    div[data-testid="stDataFrame"] { max-width:100%; overflow-x:auto; }
    @media (max-width: 900px) {
        .block-container { padding-top:1.25rem; }
        .kpi-card { min-height:112px; padding:16px; }
        .ai-result { padding:18px; }
    }
    @media (max-width: 640px) {
        .block-container { padding-left:.75rem; padding-right:.75rem; }
        .main-title { margin-top:.25rem; }
        .subtitle { margin-bottom:16px; }
        .kpi-card { min-height:100px; padding:14px; border-radius:12px; }
        .kpi-title { font-size:12px; }
        .kpi-description { font-size:11px; }
        .section-title { margin-top:18px; margin-bottom:10px; }
        .ai-result { margin-top:12px; padding:16px; border-radius:12px; }
        .ai-label { font-size:12px; }
        .stPlotlyChart { width:100% !important; }
        [data-testid="stHorizontalBlock"] { gap: .65rem; }
    }
    section[data-testid="stSidebar"] * { color: #F8FAFC; }
    .page-header { background:#EFF6FF; margin-top:40px ; border:1px solid #DBEAFE; border-radius:14px; padding:20px 22px 18px; margin-bottom:24px; }
    .main-title { font-size: 36px; font-weight: 800; color: #0F172A; margin-bottom: 4px; }
    .subtitle { color: #64748B; font-size: 16px; margin-bottom: 25px; }
    .kpi-card { background:#FFFFFF; padding:20px; min-height:132px; height:100%; box-sizing:border-box; border-radius:14px; border:1px solid #E2E8F0; box-shadow:0 2px 8px rgba(15,23,42,.04); display:flex; flex-direction:column; justify-content:center; }
    .kpi-title { color:#64748B; font-size:14px; font-weight:600; }
    .kpi-value { color:#0F172A; font-size:30px; font-weight:800; margin-top:5px; }
    .kpi-description { color:#94A3B8; font-size:12px; margin-top:4px; }
    .section-title { font-size:22px; font-weight:750; color:#0F172A; margin-top:25px; margin-bottom:15px; }
    .ai-result { padding:25px; border-radius:16px; margin-top:20px; background:#FFFFFF; border:1px solid #E2E8F0; }
    .ai-label { color:#64748B; font-size:14px; font-weight:600; }
    .ai-value { color:#0F172A; font-size:30px; font-weight:800; margin-top:5px; }
    .info-box { padding:18px; border-radius:12px; background:#FFFFFF; border:1px solid #E2E8F0; color:#475569; line-height:1.6; }
    .footer { text-align:center; color:#94A3B8; padding:30px; font-size:13px; }

    /* -------------------------------------------------------
       PRIMARY ACTION BUTTONS
       Used for: Predict Delay Risk, Fetch Current Weather,
       and Load Historical Weather.
    ------------------------------------------------------- */
    div[data-testid="stButton"] button[kind="primary"],
    div[data-testid="stFormSubmitButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #4338CA !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        min-height: 44px !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.18) !important;
    }

    div[data-testid="stButton"] button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4338CA 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border-color: #3730A3 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.30) !important;
        filter: brightness(1.04) !important;
    }

    div[data-testid="stButton"] button[kind="primary"]:active,
    div[data-testid="stFormSubmitButton"] button[kind="primary"]:active {
        transform: translateY(0) !important;
        box-shadow: 0 3px 8px rgba(37, 99, 235, 0.20) !important;
    }

    div[data-testid="stButton"] button[kind="primary"]:focus-visible,
    div[data-testid="stFormSubmitButton"] button[kind="primary"]:focus-visible {
        outline: 3px solid rgba(59, 130, 246, 0.25) !important;
        outline-offset: 2px !important;
    }

    /* Live Traffic button hover */
    div[data-testid="stButton"] button[kind="secondary"]:hover {
        background: #EFF6FF !important;
        color: #2563EB !important;
        border-color: #93C5FD !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA / MODEL
# ============================================================
@st.cache_data

def load_data():
    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        st.stop()
    return pd.read_csv(DATA_PATH)


@st.cache_resource

def load_model():
    if not MODEL_PATH.exists():
        st.error("Trained model not found. Run: python src/train_model.py")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data

def load_model_info():
    if MODEL_INFO_PATH.exists():
        return joblib.load(MODEL_INFO_PATH)
    return {}


df = load_data().copy().drop_duplicates()
model = load_model()
model_info = load_model_info()

if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Date"] = df["Timestamp"].dt.date
    df["Month"] = df["Timestamp"].dt.month
    df["Month_Name"] = df["Timestamp"].dt.strftime("%b")
    df["Hour"] = df["Timestamp"].dt.hour

if "Logistics_Delay" in df.columns:
    df["Delay_Flag"] = pd.to_numeric(df["Logistics_Delay"], errors="coerce")
else:
    df["Delay_Flag"] = np.nan

# ============================================================
# API HELPERS
# ============================================================
def get_secret(name):
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name)


WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Light freezing rain",
    67: "Heavy freezing rain", 71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
    82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


@st.cache_data(ttl=600)
def fetch_current_weather(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
        "timezone": "auto",
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    current = data.get("current", {})
    return {
        "time": current.get("time"),
        "temperature": current.get("temperature_2m"),
        "humidity": current.get("relative_humidity_2m"),
        "precipitation": current.get("precipitation"),
        "wind_speed": current.get("wind_speed_10m"),
        "weather_code": current.get("weather_code"),
        "condition": WEATHER_CODES.get(current.get("weather_code"), "Unknown"),
        "timezone": data.get("timezone", ""),
    }


@st.cache_data(ttl=3600)
def fetch_historical_weather(latitude, longitude, selected_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    day = pd.Timestamp(selected_date).strftime("%Y-%m-%d")
    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "start_date": day,
        "end_date": day,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
        "timezone": "auto",
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    hourly = data.get("hourly", {})
    if not hourly.get("time"):
        return None
    result = pd.DataFrame(hourly)
    for c in ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]:
        if c in result.columns:
            result[c] = pd.to_numeric(result[c], errors="coerce")
    return result


def weather_label(code):
    return WEATHER_CODES.get(code, "Unknown")


@st.cache_data(ttl=300)
def fetch_live_traffic(latitude, longitude, api_key):
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {
        "key": api_key,
        "point": f"{float(latitude)},{float(longitude)}",
        "unit": "kmph",
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        raise ValueError(
            f"TomTom traffic request returned HTTP {status_code}. "
            "Check the API key, coordinates, or TomTom traffic coverage."
        ) from None
    except requests.RequestException:
        raise ValueError(
            "TomTom traffic request failed. Check your internet connection or the TomTom service."
        ) from None
    data = response.json().get("flowSegmentData", {})
    current_speed = data.get("currentSpeed")
    free_flow_speed = data.get("freeFlowSpeed")
    if current_speed is None or free_flow_speed in (None, 0):
        raise ValueError("Traffic API returned incomplete speed information.")
    congestion = max(0.0, min(1.0, 1 - (float(current_speed) / float(free_flow_speed))))
    if congestion < 0.15:
        level = "Low"
    elif congestion < 0.35:
        level = "Moderate"
    elif congestion < 0.60:
        level = "High"
    else:
        level = "Severe"
    return {
        "current_speed": current_speed,
        "free_flow_speed": free_flow_speed,
        "congestion": congestion * 100,
        "level": level,
        "confidence": data.get("confidence"),
        "road_closure": data.get("roadClosure", False),
    }

# ============================================================
# UI HELPERS
# ============================================================
def kpi_card(title, value, description):
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-title">{title}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-description">{description}</div></div>',
        unsafe_allow_html=True,
    )


def section_title(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def page_header(title, subtitle):
    st.markdown(
        f'<div class="page-header"><div class="main-title">{title}</div>'
        f'<div class="subtitle">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def delay_percentage(frame):
    if "Delay_Flag" not in frame.columns or frame["Delay_Flag"].notna().sum() == 0:
        return 0.0
    return float(frame["Delay_Flag"].mean() * 100)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        '<div style="font-size:25px;font-weight:800;margin-bottom:5px;">🚚 Shipment AI</div>'
        '<div style="color:#94A3B8;font-size:13px;margin-bottom:25px;">Intelligent Logistics Analytics</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "📊 Executive Dashboard",
            "🔮 Delay Predictor",
            "📈 Shipment Analytics",
            "🚦 Traffic & Route",
            "🌦 Weather Analysis",
            "🏭 Supplier / Asset Performance",
            "🤖 AI Insights",
        ],
    )


# ============================================================
# PAGE 1 — EXECUTIVE DASHBOARD
# ============================================================
if page == "📊 Executive Dashboard":
    page_header("🚚 Shipment Delay Intelligence", "AI-powered logistics monitoring and shipment risk analytics")

    total_shipments = len(df)
    delayed_shipments = int(df["Delay_Flag"].fillna(0).sum())
    delay_rate = delay_percentage(df)
    avg_waiting = float(df["Waiting_Time"].mean()) if "Waiting_Time" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("TOTAL SHIPMENTS", f"{total_shipments:,}", "Records analyzed")
    with c2: kpi_card("DELAYED SHIPMENTS", f"{delayed_shipments:,}", "Recorded delayed shipments")
    with c3: kpi_card("DELAY RATE", f"{delay_rate:.1f}%", "Observed dataset rate")
    with c4: kpi_card("AVG WAITING TIME", f"{avg_waiting:.1f}", "Average waiting time")

    section_title("Shipment Performance Overview")
    c1, c2 = st.columns(2)
    with c1:
        if "Shipment_Status" in df.columns:
            counts = df["Shipment_Status"].value_counts().reset_index()
            counts.columns = ["Status", "Count"]
            fig = px.pie(counts, names="Status", values="Count", hole=0.55, title="Shipment Status")
            fig.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "Traffic_Status" in df.columns:
            counts = df["Traffic_Status"].value_counts().reset_index()
            counts.columns = ["Traffic", "Shipments"]
            fig = px.bar(counts, x="Traffic", y="Shipments", title="Shipments by Traffic Condition")
            fig.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)

    section_title("Delay Analysis")
    if "Traffic_Status" in df.columns:
        traffic_delay = df.groupby("Traffic_Status")["Delay_Flag"].mean().reset_index()
        traffic_delay["Delay Rate"] = traffic_delay["Delay_Flag"] * 100
        fig = px.bar(traffic_delay, x="Traffic_Status", y="Delay Rate", title="Observed Delay Rate by Traffic Condition",
                     labels={"Traffic_Status":"Traffic Condition", "Delay Rate":"Delay Rate (%)"})
        fig.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

    section_title("Recent Shipment Records")
    display_columns = [c for c in ["Timestamp","Asset_ID","Shipment_Status","Traffic_Status","Temperature","Humidity","Waiting_Time","Logistics_Delay"] if c in df.columns]
    st.dataframe(
        df[display_columns].head(15),
        use_container_width=True,
        hide_index=True,
        column_config={c: st.column_config.Column(width="small") for c in display_columns},
    )

# ============================================================
# PAGE 2 — DELAY PREDICTOR
# ============================================================
elif page == "🔮 Delay Predictor":
    page_header("🔮 AI Shipment Delay Predictor", "Enter shipment conditions to estimate delay risk.")
    section_title("Optional Live Weather")
    wc1, wc2, wc3 = st.columns(3, vertical_alignment="top")

    # Equal-height weather input cards
    with wc1:
        with st.container(border=True, height=105):
            weather_lat = st.number_input(
                "Weather Latitude",
                value=20.5,
                min_value=-90.0,
                max_value=90.0,
                key="weather_lat_pred",
            )

    with wc2:
        with st.container(border=True, height=105):
            weather_lon = st.number_input(
                "Weather Longitude",
                value=72.9,
                min_value=-180.0,
                max_value=180.0,
                key="weather_lon_pred",
            )

    with wc3:
        with st.container(border=True, height=105):
            st.write("")
            fetch_weather = st.button(
                "🌦 Fetch Current Weather",
                use_container_width=True,
                key="fetch_current_weather_pred",
                type="primary",
            )

    # Show the API response below the row so it never changes the card heights.
    if fetch_weather:
        try:
            live = fetch_current_weather(weather_lat, weather_lon)
            st.session_state["pred_temperature"] = float(live["temperature"])
            st.session_state["pred_humidity"] = float(live["humidity"])
            st.success(
                f"{live['condition']} • {live['temperature']} °C • "
                f"{live['humidity']}% humidity"
            )
        except Exception as error:
            st.error(f"Weather request failed: {error}")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            asset_id = st.text_input("Asset / Supplier ID", value="Truck_1")
            latitude = st.number_input("Latitude", value=20.5, min_value=-90.0, max_value=90.0)
            longitude = st.number_input("Longitude", value=72.9, min_value=-180.0, max_value=180.0)
            inventory = st.number_input("Inventory Level", min_value=0.0, value=500.0)
            asset_utilization = st.number_input("Asset Utilization (%)", min_value=0.0, max_value=100.0, value=80.0)
            demand_forecast = st.number_input("Demand Forecast", min_value=0.0, value=200.0)
        with c2:
            temperature = st.number_input("Temperature (°C)", value=float(st.session_state.get("pred_temperature", 30.0)))
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=float(st.session_state.get("pred_humidity", 70.0)))
            waiting_time = st.number_input("Waiting Time", min_value=0.0, value=40.0)
            transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=5000.0)
            purchase_frequency = st.number_input("Purchase Frequency", min_value=0.0, value=5.0)
            prediction_date = st.date_input("Shipment Date", value=date.today())

        predict_button = st.form_submit_button(
            "🔮 PREDICT DELAY RISK",
            use_container_width=True,
            type="primary",
        )

    if predict_button:
        date_obj = pd.Timestamp(prediction_date)
        day_of_week = date_obj.dayofweek
        is_weekend = int(day_of_week >= 5)

        temperature_risk = "Low" if temperature <= 10 else "Normal" if temperature <= 25 else "High" if temperature <= 35 else "Extreme"
        humidity_risk = "Low" if humidity <= 30 else "Normal" if humidity <= 60 else "High" if humidity <= 80 else "Very High"
        waiting_risk = "Low" if waiting_time <= 20 else "Medium" if waiting_time <= 40 else "High" if waiting_time <= 60 else "Critical"
        inventory_risk = "Low" if inventory <= 250 else "Medium" if inventory <= 500 else "High" if inventory <= 750 else "Very High"

        input_data = {
            "Asset_ID": asset_id,
            "Latitude": latitude,
            "Longitude": longitude,
            "Inventory_Level": inventory,
            "Temperature": temperature,
            "Humidity": humidity,
            "Waiting_Time": waiting_time,
            "User_Transaction_Amount": transaction_amount,
            "User_Purchase_Frequency": purchase_frequency,
            "Asset_Utilization": asset_utilization,
            "Demand_Forecast": demand_forecast,
            "Year": date_obj.year,
            "Month": date_obj.month,
            "Day": date_obj.day,
            "Hour": 12,
            "Day_of_Week": day_of_week,
            "Is_Weekend": is_weekend,
            "Temperature_Risk": temperature_risk,
            "Humidity_Risk": humidity_risk,
            "Waiting_Risk": waiting_risk,
            "Inventory_Risk": inventory_risk,
        }

        try:
            input_df = pd.DataFrame([input_data])
            prediction = int(model.predict(input_df)[0])
            probability = float(model.predict_proba(input_df)[0][1])

            if prediction == 1:
                status = "⚠️ HIGH DELAY RISK"
                recommendation = "Monitor this shipment closely and review operational conditions before dispatch."
            else:
                status = "✅ LOWER DELAY RISK"
                recommendation = "The model estimates a lower probability of delay for the supplied conditions."

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="ai-result"><div class="ai-label">AI PREDICTION</div><div class="ai-value">{status}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="ai-result"><div class="ai-label">DELAY PROBABILITY</div><div class="ai-value">{probability*100:.1f}%</div></div>', unsafe_allow_html=True)

            st.progress(probability, text=f"Estimated delay probability: {probability*100:.1f}%")
            st.info(f"💡 Recommendation: {recommendation}")

            section_title("Prediction Input Summary")
            summary = pd.DataFrame({
                "Feature": ["Temperature","Humidity","Waiting Time","Inventory Level","Asset Utilization","Demand Forecast","Temperature Risk","Humidity Risk","Waiting Risk","Inventory Risk"],
                "Value": [temperature,humidity,waiting_time,inventory,asset_utilization,demand_forecast,temperature_risk,humidity_risk,waiting_risk,inventory_risk],
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)
        except Exception as error:
            st.error("Prediction could not be completed.")
            st.exception(error)

# ============================================================
# PAGE 3 — SHIPMENT ANALYTICS
# ============================================================
elif page == "📈 Shipment Analytics":
    page_header("📈 Shipment Analytics", "Explore shipment behavior and logistics performance.")
    filtered_df = df.copy()
    c1, c2 = st.columns(2)
    with c1:
        if "Shipment_Status" in df.columns:
            statuses = sorted(df["Shipment_Status"].dropna().unique().tolist())
            selected = st.multiselect("Shipment Status", statuses, default=statuses)
            filtered_df = filtered_df[filtered_df["Shipment_Status"].isin(selected)]
    with c2:
        if "Traffic_Status" in df.columns:
            traffic = sorted(df["Traffic_Status"].dropna().unique().tolist())
            selected = st.multiselect("Traffic Status", traffic, default=traffic)
            filtered_df = filtered_df[filtered_df["Traffic_Status"].isin(selected)]

    c1, c2 = st.columns(2)
    with c1:
        if "Temperature" in filtered_df.columns:
            fig = px.histogram(filtered_df, x="Temperature", nbins=25, title="Temperature Distribution")
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "Waiting_Time" in filtered_df.columns:
            fig = px.histogram(filtered_df, x="Waiting_Time", nbins=25, title="Waiting Time Distribution")
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    if "Temperature" in filtered_df.columns and "Waiting_Time" in filtered_df.columns:
        fig = px.scatter(filtered_df, x="Temperature", y="Waiting_Time",
                         color="Shipment_Status" if "Shipment_Status" in filtered_df.columns else None,
                         title="Temperature vs Operational Waiting")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 4 — TRAFFIC & ROUTE
# ============================================================
elif page == "🚦 Traffic & Route":
    page_header("🚦 Traffic & Route Analysis", "Analyze shipment locations, historical traffic conditions, and optional live traffic flow.")

    if "Latitude" in df.columns and "Longitude" in df.columns:
        section_title("Shipment Location Map")
        map_df = df[["Latitude", "Longitude"]].dropna().rename(columns={"Latitude":"lat", "Longitude":"lon"})
        st.map(map_df, use_container_width=True)
        st.caption("The dataset contains shipment coordinates. The map shows shipment locations; it is not a road-by-road route reconstruction.")

    if "Traffic_Status" in df.columns:
        section_title("Historical / Dataset Traffic")
        counts = df["Traffic_Status"].value_counts().reset_index()
        counts.columns = ["Traffic Status", "Shipments"]
        fig = px.bar(counts, x="Traffic Status", y="Shipments", title="Shipment Count by Traffic Condition")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        if "Waiting_Time" in df.columns:
            fig = px.box(df, x="Traffic_Status", y="Waiting_Time", title="Waiting Time by Traffic Condition")
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    section_title("Live Traffic")
    tomtom_key = get_secret("TOMTOM_API_KEY") or ""

    if "Latitude" in df.columns and "Longitude" in df.columns:
        default_live_lat = 23.0225
        default_live_lon = 72.5714
        st.markdown("**Live Traffic Location**")
        c1, c2, c3 = st.columns([1, 1, 1], gap="medium")
        with c1:
            live_lat = st.number_input("Latitude", value=default_live_lat, min_value=-90.0, max_value=90.0, key="live_lat")
        with c2:
            live_lon = st.number_input("Longitude", value=default_live_lon, min_value=-180.0, max_value=180.0, key="live_lon")
        with c3:
            st.write("")
            fetch_traffic = st.button("🚦 Check Live Traffic", use_container_width=True)

        if fetch_traffic:
            if not tomtom_key:
                st.warning("No TomTom API key was entered. The live API cannot be called, but your dataset-based traffic analysis is already available above.")
                st.markdown("**To enable real-time traffic:** create a TomTom developer account, generate an API key, then paste it into the field above. Do not publish the key in GitHub.")
            else:
                try:
                    traffic_data = fetch_live_traffic(live_lat, live_lon, tomtom_key)
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: kpi_card("CURRENT SPEED", f"{traffic_data['current_speed']:.0f} km/h", "TomTom")
                    with c2: kpi_card("FREE-FLOW SPEED", f"{traffic_data['free_flow_speed']:.0f} km/h", "Reference speed")
                    with c3: kpi_card("CONGESTION", f"{traffic_data['congestion']:.1f}%", traffic_data["level"])
                    with c4: kpi_card("CONFIDENCE", f"{traffic_data['confidence']*100:.0f}%" if traffic_data['confidence'] is not None else "N/A", "Traffic data confidence")
                except Exception as error:
                    st.error(str(error))

# ============================================================
# PAGE 5 — WEATHER
# ============================================================
elif page == "🌦 Weather Analysis":
    page_header("🌦 Weather & Environmental Analysis", "Combine historical shipment weather fields with live weather data from Open-Meteo.")

    c1, c2 = st.columns(2)
    with c1:
        if "Temperature" in df.columns:
            fig = px.histogram(df, x="Temperature", nbins=30, title="Dataset Temperature Distribution")
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "Humidity" in df.columns:
            fig = px.histogram(df, x="Humidity", nbins=30, title="Dataset Humidity Distribution")
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    if "Temperature" in df.columns and "Humidity" in df.columns:
        fig = px.scatter(df, x="Temperature", y="Humidity",
                         color="Traffic_Status" if "Traffic_Status" in df.columns else None,
                         title="Dataset Temperature vs Humidity")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    section_title("Live Weather by Shipment Location")
    if "Latitude" in df.columns and "Longitude" in df.columns:
        valid_locations = df[["Asset_ID","Latitude","Longitude"]].dropna().copy()
        valid_locations = valid_locations.reset_index(drop=True)
        selected_idx = st.selectbox("Select shipment location", valid_locations.index, format_func=lambda i: f"{valid_locations.loc[i, 'Asset_ID']} — {valid_locations.loc[i, 'Latitude']:.4f}, {valid_locations.loc[i, 'Longitude']:.4f}")
        row = valid_locations.loc[selected_idx]
        try:
            live = fetch_current_weather(row["Latitude"], row["Longitude"])
            c1, c2, c3, c4 = st.columns(4)
            with c1: kpi_card("TEMPERATURE", f"{live['temperature']:.1f} °C", live["condition"])
            with c2: kpi_card("HUMIDITY", f"{live['humidity']:.0f}%", "Current")
            with c3: kpi_card("PRECIPITATION", f"{live['precipitation']:.1f} mm", "Current")
            with c4: kpi_card("WIND", f"{live['wind_speed']:.1f} km/h", "Current")
            st.caption(f"Weather time: {live['time']} • Timezone: {live['timezone']}")
        except Exception as error:
            st.error(f"Live weather request failed: {error}")

    section_title("Historical Weather for a Shipment Date")
    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        hist_lat = st.number_input("Historical Latitude", value=20.5, min_value=-90.0, max_value=90.0)
    with hc2:
        hist_lon = st.number_input("Historical Longitude", value=72.9, min_value=-180.0, max_value=180.0)
    with hc3:
        hist_date = st.date_input("Historical Date", value=date(2024, 7, 1))

    if st.button(
        "📅 Load Historical Weather",
        use_container_width=True,
        type="primary",
    ):
        try:
            historical = fetch_historical_weather(hist_lat, hist_lon, hist_date)
            if historical is None:
                st.warning("No historical weather data was returned.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                with c1: kpi_card("AVG TEMP", f"{historical['temperature_2m'].mean():.1f} °C", "Selected date")
                with c2: kpi_card("AVG HUMIDITY", f"{historical['relative_humidity_2m'].mean():.0f}%", "Selected date")
                with c3: kpi_card("TOTAL PRECIP.", f"{historical['precipitation'].sum():.1f} mm", "Selected date")
                with c4: kpi_card("AVG WIND", f"{historical['wind_speed_10m'].mean():.1f} km/h", "Selected date")
                chart_df = historical[["time","temperature_2m","relative_humidity_2m"]].copy()
                fig = px.line(chart_df, x="time", y=["temperature_2m","relative_humidity_2m"], title="Historical Hourly Weather")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as error:
            st.error(f"Historical weather request failed: {error}")

# ============================================================
# PAGE 6 — SUPPLIER / ASSET PERFORMANCE
# ============================================================
elif page == "🏭 Supplier / Asset Performance":
    page_header("🏭 Supplier / Asset Performance", "Compare logistics performance across assets.")
    if "Asset_ID" in df.columns:
        asset_summary = df.groupby("Asset_ID").agg(
            Shipments=("Asset_ID", "count"),
            Average_Waiting_Time=("Waiting_Time", "mean"),
        ).reset_index()
        delay_summary = df.groupby("Asset_ID")["Delay_Flag"].mean().reset_index()
        delay_summary["Delay Rate"] = delay_summary["Delay_Flag"] * 100
        asset_summary = asset_summary.merge(delay_summary[["Asset_ID","Delay Rate"]], on="Asset_ID", how="left")
        st.dataframe(
            asset_summary.sort_values("Shipments", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Asset_ID": st.column_config.Column(width="small"),
                "Shipments": st.column_config.Column(width="small"),
                "Average_Waiting_Time": st.column_config.Column(width="medium"),
                "Delay Rate": st.column_config.Column(width="small"),
            },
        )
        fig = px.bar(asset_summary.sort_values("Delay Rate", ascending=False).head(20), x="Asset_ID", y="Delay Rate", title="Assets by Observed Delay Rate")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 7 — AI INSIGHTS
# ============================================================
elif page == "🤖 AI Insights":
    page_header("🤖 AI-Generated Logistics Insights", "Model metrics and data-driven observations.")
    section_title("Model Information")
    if model_info:
        metrics = model_info.get("metrics", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("MODEL", model_info.get("model_name","Unknown"), "Selected using test-set F1")
        with c2: kpi_card("ACCURACY", f"{metrics.get('accuracy',0)*100:.1f}%", "Held-out test set")
        with c3: kpi_card("PRECISION", f"{metrics.get('precision',0)*100:.1f}%", "Delay class")
        with c4: kpi_card("F1 SCORE", f"{metrics.get('f1',0)*100:.1f}%", "Balanced metric")
        st.caption("These metrics come from a single 80/20 stratified split of the supplied dataset. They are not evidence of production-level model performance.")

    section_title("Key Data Insights")
    insights = []
    if "Traffic_Status" in df.columns:
        traffic_delay = df.groupby("Traffic_Status")["Delay_Flag"].mean().sort_values(ascending=False)
        if len(traffic_delay):
            insights.append(f"🚦 Traffic: {traffic_delay.index[0]} has the highest observed delay rate ({traffic_delay.iloc[0]*100:.1f}%) in this dataset.")
    if "Waiting_Time" in df.columns:
        insights.append(f"⏱️ Waiting Time: average operational waiting time is {df['Waiting_Time'].mean():.1f}.")
    if "Temperature" in df.columns:
        insights.append(f"🌡️ Temperature: observed values range from {df['Temperature'].min():.1f} to {df['Temperature'].max():.1f} °C.")
    if "Humidity" in df.columns:
        insights.append(f"💧 Humidity: average recorded humidity is {df['Humidity'].mean():.1f}%.")
    for insight in insights:
        st.markdown(f'<div class="info-box">{insight}</div>', unsafe_allow_html=True)
        st.write("")
    st.warning("Dataset insights describe associations. They should not be interpreted as proof that a factor causes shipment delays.")
