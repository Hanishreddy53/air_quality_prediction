# ============================================================
# AIR QUALITY PREDICTION SYSTEM
# WEEK 8 - STREAMLIT DASHBOARD
# ============================================================

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from week7_explainable_ai_alerts import (
    get_aqi_category,
    get_health_recommendation,
    generate_alert,
    prepare_features,
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Air Quality AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Main content */
    .main {
        padding-top: 1rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Main title */
    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #64748b;
        margin-bottom: 25px;
    }

    /* Dashboard cards */
    .dashboard-card {
        background-color: white;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 18px;
    }

    .card-title {
        font-size: 15px;
        color: #64748b;
        margin-bottom: 8px;
    }

    .card-value {
        font-size: 28px;
        font-weight: 700;
    }

    /* Section heading */
    .section-title {
        font-size: 24px;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    /* Info box */
    .info-box {
        background-color: white;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(
            135deg,
            #0f766e,
            #155e75
        );
        padding: 35px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 38px;
        margin-bottom: 10px;
    }

    .hero p {
        font-size: 17px;
        opacity: 0.9;
        max-width: 850px;
    }

    /* Small text */
    .small-text {
        color: #64748b;
        font-size: 14px;
    }

    /* Explicit high-contrast text for light main content */
    .stApp,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6,
    .stApp p,
    .stApp label,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricValue"] {
        color: #0f172a !important;
    }

    .stApp input,
    .stApp textarea,
    .stApp select {
        color: #0f172a !important;
        background-color: #ffffff !important;
    }

    .hero,
    .hero h1,
    .hero p,
    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] label {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

DATA_PATH = ROOT_DIR / "data" / "processed" / "feature_engineered_dataset.csv"

RF_MODEL_PATH = ROOT_DIR / "models" / "optimized_random_forest_regressor.pkl"

XGB_MODEL_PATH = ROOT_DIR / "models" / "optimized_xgboost_classifier.pkl"

RF_IMPORTANCE_PATH = ROOT_DIR / "outputs" / "reports" / "regression_feature_importance.csv"

XGB_IMPORTANCE_PATH = ROOT_DIR / "outputs" / "reports" / "classification_feature_importance.csv"

ALERT_RESULTS_PATH = ROOT_DIR / "outputs" / "reports" / "week7_prediction_alert_results.csv"

PLOT_DIR = ROOT_DIR / "outputs" / "plots" / "week7"


# ============================================================
# 4. LOAD DATA
# ============================================================

@st.cache_data
def load_dataset():

    if not os.path.exists(DATA_PATH):
        return None

    return pd.read_csv(DATA_PATH)


# ============================================================
# 5. LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    rf_model = None
    xgb_model = None

    if os.path.exists(RF_MODEL_PATH):

        rf_model = joblib.load(
            RF_MODEL_PATH
        )

    if os.path.exists(XGB_MODEL_PATH):

        xgb_model = joblib.load(
            XGB_MODEL_PATH
        )

    return rf_model, xgb_model


# ============================================================
# 6. LOAD WEEK 7 OUTPUTS
# ============================================================

@st.cache_data
def load_week7_results():

    rf_importance = None
    xgb_importance = None
    alert_results = None

    if os.path.exists(
        RF_IMPORTANCE_PATH
    ):

        rf_importance = pd.read_csv(
            RF_IMPORTANCE_PATH
        )

    if os.path.exists(
        XGB_IMPORTANCE_PATH
    ):

        xgb_importance = pd.read_csv(
            XGB_IMPORTANCE_PATH
        )

    if os.path.exists(
        ALERT_RESULTS_PATH
    ):

        alert_results = pd.read_csv(
            ALERT_RESULTS_PATH
        )

    return (
        rf_importance,
        xgb_importance,
        alert_results
    )


# ============================================================
# 7. INITIALIZE
# ============================================================

loading_message = st.empty()

loading_message.info(
    "Loading the air-quality dataset and trained models. "
    "The first load may take a few seconds..."
)

df = load_dataset()

rf_model, xgb_model = load_models()

(
    rf_importance,
    xgb_importance,
    alert_results
) = load_week7_results()

loading_message.empty()


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:10px 0 20px 0;
        ">
            <div style="font-size:42px;">🌍</div>
            <h2 style="margin:0;">Air Quality AI</h2>
            <p style="
                color:#cbd5e1;
                font-size:13px;
            ">
                ML Monitoring System
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🔮 Prediction",
            "🚨 Health Alerts"
        ]
    )

    st.divider()

    st.caption(
        "Air Quality Prediction System"
    )

    st.caption(
        "Machine Learning + Explainable AI"
    )


# ============================================================
# 9. HELPER — PREPARE MODEL INPUT
# ============================================================

def prepare_prediction_data(
    selected_data,
    model
):
    return prepare_features(selected_data, model)


def calculate_aqi_sub_index(concentration, breakpoints):
    """Calculate one CPCB AQI sub-index for manually entered measurements."""
    for concentration_low, concentration_high, index_low, index_high in breakpoints:
        if concentration_low <= concentration <= concentration_high:
            return ((index_high - index_low) / (concentration_high - concentration_low)) * (
                concentration - concentration_low
            ) + index_low
    return 500.0 if concentration > breakpoints[-1][1] else 0.0


def calculate_manual_aqi(values):
    """Create the AQI-derived fields expected by the Week 6 model."""
    breakpoints = {
        "PM2.5": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)],
        "PM10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)],
        "NO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), (401, 800, 401, 500)],
        "SO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2000, 401, 500)],
        "CO": [(0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200), (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 50, 401, 500)],
        "O3": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)],
    }
    indices = {
        "AQI_PM2.5": calculate_aqi_sub_index(values["PM2.5"], breakpoints["PM2.5"]),
        "AQI_PM10": calculate_aqi_sub_index(values["PM10"], breakpoints["PM10"]),
        "AQI_NO2": calculate_aqi_sub_index(values["NO2"], breakpoints["NO2"]),
        "AQI_SO2": calculate_aqi_sub_index(values["SO2"], breakpoints["SO2"]),
        "AQI_CO": calculate_aqi_sub_index(values["CO"], breakpoints["CO"]),
        "AQI_O3": calculate_aqi_sub_index(values["O3"], breakpoints["O3"]),
    }
    indices["AQI"] = min(500.0, max(indices.values()))
    if indices["AQI"] <= 50:
        category = "Good"
    elif indices["AQI"] <= 100:
        category = "Satisfactory"
    elif indices["AQI"] <= 200:
        category = "Moderately Polluted"
    elif indices["AQI"] <= 300:
        category = "Poor"
    elif indices["AQI"] <= 400:
        category = "Very Poor"
    else:
        category = "Severe"
    indices["AQI_Category"] = category
    indices["AQI_Class"] = {"Good": 0, "Moderately Polluted": 1, "Poor": 2, "Satisfactory": 3, "Severe": 4, "Very Poor": 5}[category]
    return indices


# ============================================================
# PAGE 1 — HOME
# ============================================================

if page == "🏠 Home":

    st.markdown(
        """
        <div class="hero">

        <h1>Air Quality Prediction System</h1>

        <p>
        An end-to-end machine learning system for predicting
        PM2.5 concentration, analyzing pollution patterns,
        explaining model decisions and generating health alerts.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ----------------------------------------
    # Overview
    # ----------------------------------------

    st.markdown(
        '<div class="section-title">'
        'System Overview'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">

        <p>
        The system uses historical air-quality and meteorological
        data collected from multiple monitoring stations.
        Machine learning models are used to predict pollution
        levels and classify air-quality conditions.
        </p>

        <p>
        The dashboard also provides model explainability,
        pollution analytics and health-oriented alerts.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ----------------------------------------
    # Metrics
    # ----------------------------------------

    if df is not None:

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Records",
                f"{len(df):,}"
            )

        with col2:

            st.metric(
                "Features",
                len(df.columns)
            )

        with col3:

            if "station" in df.columns:

                station_count = (
                    df["station"].nunique()
                )

            else:

                station_count = 12

            st.metric(
                "Stations",
                station_count
            )

        with col4:

            if "PM2.5" in df.columns:

                avg_pm25 = (
                    df["PM2.5"].mean()
                )

                st.metric(
                    "Avg PM2.5",
                    f"{avg_pm25:.2f}"
                )

    st.divider()

    # ----------------------------------------
    # Models
    # ----------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Machine Learning Models'
        '</div>',
        unsafe_allow_html=True
    )

    model_col1, model_col2 = st.columns(2)

    with model_col1:

        st.markdown(
            """
            <div class="dashboard-card">

            <div class="card-title">
            REGRESSION MODEL
            </div>

            <div class="card-value">
            🌲 Random Forest
            </div>

            <p class="small-text">
            Predicts numerical PM2.5 concentration.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        if rf_model is not None:

            st.success(
                "Model loaded successfully"
            )

        else:

            st.error(
                "Model file not found"
            )

    with model_col2:

        st.markdown(
            """
            <div class="dashboard-card">

            <div class="card-title">
            CLASSIFICATION MODEL
            </div>

            <div class="card-value">
            ⚡ XGBoost
            </div>

            <p class="small-text">
            Predicts air-quality categories.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        if xgb_model is not None:

            st.success(
                "Model loaded successfully"
            )

        else:

            st.error(
                "Model file not found"
            )

    st.divider()

    st.info(
        "Use the navigation panel to explore "
        "predictions, analytics, explainability "
        "and health alerts."
    )


# ============================================================
# PAGE 2 — PREDICTION
# ============================================================

elif page == "🔮 Prediction":

    st.markdown(
        '<div class="main-title">'
        'Air Quality Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter current pollution and weather readings to predict PM2.5 and receive an instant health alert.'
        '</div>',
        unsafe_allow_html=True
    )

    if df is None:

        st.error(
            "Feature-engineered dataset not found."
        )

        st.stop()

    if rf_model is None:

        st.error(
            "Random Forest model not found."
        )

        st.stop()

    st.markdown(
        '<div class="section-title">'
        'Enter Air Quality Measurements'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Enter pollutant, weather, time, and season values. The model predicts PM2.5 and the entered readings determine the AQI alert."
    )

    with st.form("air_quality_prediction_form"):
        pollutant_col1, pollutant_col2, pollutant_col3 = st.columns(3)

        with pollutant_col1:
            pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=30.0, step=1.0)
            pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=80.0, step=1.0)

        with pollutant_col2:
            so2 = st.number_input("SO₂ (µg/m³)", min_value=0.0, value=15.0, step=1.0)
            no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, value=45.0, step=1.0)
            co = st.number_input("CO (mg/m³)", min_value=0.0, value=1.0, step=0.1)

        with pollutant_col3:
            o3 = st.number_input("O₃ (µg/m³)", min_value=0.0, value=45.0, step=1.0)
            temp = st.number_input("TEMP (°C)", value=15.0, step=0.5)

        st.markdown("**Weather readings**")
        weather_col1, weather_col2, weather_col3 = st.columns(3)

        with weather_col1:
            pressure = st.number_input("PRES (hPa)", min_value=800.0, value=1010.0, step=1.0)

        with weather_col2:
            dew_point = st.number_input("DEWP (°C)", value=8.0, step=0.5)

        with weather_col3:
            rain = st.number_input("RAIN (mm)", min_value=0.0, value=0.0, step=0.1)

        wind_speed = st.number_input("WSPM (m/s)", min_value=0.0, value=2.0, step=0.1)

        time_col1, time_col2, time_col3 = st.columns(3)
        with time_col1:
            hour = st.number_input("Hour", min_value=0, max_value=23, value=12, step=1)
        with time_col2:
            month = st.number_input("Month", min_value=1, max_value=12, value=6, step=1)
        with time_col3:
            season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"], index=1)

        submitted = st.form_submit_button(
            "🔮 Predict Air Quality",
            type="primary",
            use_container_width=True
        )

    if submitted:

        try:

            # Use a real processed row to preserve the engineered feature
            # schema, then replace it with the user's current measurements.
            selected_row = df.iloc[[0]].copy()
            user_values = {
                "PM2.5": pm25,
                "PM10": pm10,
                "SO2": so2,
                "NO2": no2,
                # Week 4 stores CO in µg/m³; the form accepts mg/m³.
                "CO": co * 1000,
                "O3": o3,
                "TEMP": temp,
                "PRES": pressure,
                "DEWP": dew_point,
                "RAIN": rain,
                "WSPM": wind_speed,
                "hour": hour,
                "Hour": hour,
                "month": month,
                "Month": month,
                "Season": season,
            }
            for column, value in user_values.items():
                if column in selected_row.columns:
                    selected_row[column] = value

            manual_aqi = calculate_manual_aqi({
                "PM2.5": pm25,
                "PM10": pm10,
                "SO2": so2,
                "NO2": no2,
                "CO": co,
                "O3": o3,
            })
            for column, value in manual_aqi.items():
                if column in selected_row.columns:
                    selected_row[column] = value

            X_selected = prepare_prediction_data(selected_row, rf_model)

            predicted_pm25 = (
                rf_model.predict(
                    X_selected
                )[0]
            )

            predicted_pm25 = float(
                predicted_pm25
            )

            # The optimized Week 6 model can contain AQI-derived columns.
            # For manual readings, use the freshly calculated input AQI so
            # the category and alert always reflect what the user entered.
            category = manual_aqi["AQI_Category"]

            alert = (
                generate_alert(
                    category
                )
            )

            recommendation = (
                get_health_recommendation(
                    category
                )
            )

            # --------------------------------
            # Results
            # --------------------------------

            st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Predicted PM2.5",
                    f"{predicted_pm25:.2f}"
                )

            with col2:

                st.metric(
                    "Air Quality",
                    category
                )

            with col3:

                st.metric(
                    "Alert Level",
                    alert
                )

            with col4:

                st.metric(
                    "Input AQI",
                    f"{manual_aqi['AQI']:.0f}"
                )

            st.divider()

            st.markdown(
                '<div class="section-title">'
                'Health Recommendation'
                '</div>',
                unsafe_allow_html=True
            )

            st.info(
                recommendation
            )

        except Exception as error:

            st.error(
                "Prediction could not be generated."
            )

            st.exception(error)


# ============================================================
# PAGE 3 — ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.markdown(
        '<div class="main-title">'
        'Air Quality Analytics'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore pollution statistics and monitoring-station patterns.'
        '</div>',
        unsafe_allow_html=True
    )

    if df is None:

        st.error(
            "Dataset not found."
        )

        st.stop()

    pollutants = [
        column
        for column in [
            "PM2.5",
            "PM10",
            "SO2",
            "NO2",
            "CO",
            "O3"
        ]
        if column in df.columns
    ]

    # ----------------------------------------
    # Statistics
    # ----------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Pollutant Statistics'
        '</div>',
        unsafe_allow_html=True
    )

    if pollutants:

        statistics = (
            df[pollutants]
            .describe()
            .T
            .round(2)
        )

        st.dataframe(
            statistics,
            use_container_width=True
        )

    st.divider()

    # ----------------------------------------
    # PM2.5 distribution
    # ----------------------------------------

    if "PM2.5" in df.columns:

        st.markdown(
            '<div class="section-title">'
            'PM2.5 Distribution'
            '</div>',
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        ax.hist(
            df["PM2.5"].dropna(),
            bins=50
        )

        ax.set_xlabel(
            "PM2.5 Concentration"
        )

        ax.set_ylabel(
            "Frequency"
        )

        ax.set_title(
            "Distribution of PM2.5"
        )

        ax.grid(
            alpha=0.2
        )

        st.pyplot(fig)

        plt.close(fig)

    # ----------------------------------------
    # Average pollutants
    # ----------------------------------------

    if pollutants:

        st.markdown(
            '<div class="section-title">'
            'Average Pollutant Concentrations'
            '</div>',
            unsafe_allow_html=True
        )

        averages = (
            df[pollutants]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        ax.bar(
            averages.index,
            averages.values
        )

        ax.set_ylabel(
            "Average Concentration"
        )

        ax.set_xlabel(
            "Pollutant"
        )

        ax.set_title(
            "Average Pollutant Concentrations"
        )

        ax.grid(
            axis="y",
            alpha=0.2
        )

        st.pyplot(fig)

        plt.close(fig)

    # ----------------------------------------
    # Station analysis
    # ----------------------------------------

    if (
        "station" in df.columns
        and
        "PM2.5" in df.columns
    ):

        st.markdown(
            '<div class="section-title">'
            'Station-wise PM2.5'
            '</div>',
            unsafe_allow_html=True
        )

        station_values = (
            df.groupby(
                "station"
            )["PM2.5"]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        ax.bar(
            station_values.index,
            station_values.values
        )

        ax.set_ylabel(
            "Average PM2.5"
        )

        ax.set_xlabel(
            "Monitoring Station"
        )

        ax.set_title(
            "Average PM2.5 by Station"
        )

        plt.xticks(
            rotation=70
        )

        ax.grid(
            axis="y",
            alpha=0.2
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


# ============================================================
# PAGE 4 — EXPLAINABILITY
# ============================================================

elif page == "🧠 Explainability":

    st.markdown(
        '<div class="main-title">'
        'Model Explainability'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Understand which features influenced the trained models.'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "These results were generated during Week 7. "
        "The dashboard only displays the saved SHAP and "
        "feature-importance results."
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🌲 Random Forest — PM2.5 Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    rf_tab1, rf_tab2 = st.tabs(
        [
            "Feature Importance",
            "SHAP Analysis"
        ]
    )

    with rf_tab1:

        if rf_importance is not None:

            st.dataframe(
                rf_importance.head(15),
                use_container_width=True,
                hide_index=True
            )

        feature_plot = os.path.join(
            PLOT_DIR,
            "random_forest_feature_importance.png"
        )

        if os.path.exists(
            feature_plot
        ):

            st.image(
                feature_plot,
                use_container_width=True
            )

    with rf_tab2:

        shap_summary = os.path.join(
            PLOT_DIR,
            "random_forest_shap_summary.png"
        )

        shap_bar = os.path.join(
            PLOT_DIR,
            "random_forest_shap_bar.png"
        )

        if os.path.exists(
            shap_summary
        ):

            st.image(
                shap_summary,
                use_container_width=True,
                caption="SHAP Summary"
            )

        if os.path.exists(
            shap_bar
        ):

            st.image(
                shap_bar,
                use_container_width=True,
                caption="SHAP Feature Importance"
            )

    st.divider()

    # ========================================================
    # XGBOOST
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '⚡ XGBoost — Air Quality Classification'
        '</div>',
        unsafe_allow_html=True
    )

    xgb_tab1, xgb_tab2 = st.tabs(
        [
            "Feature Importance",
            "SHAP Analysis"
        ]
    )

    with xgb_tab1:

        if xgb_importance is not None:

            st.dataframe(
                xgb_importance.head(15),
                use_container_width=True,
                hide_index=True
            )

        feature_plot = os.path.join(
            PLOT_DIR,
            "xgboost_feature_importance.png"
        )

        if os.path.exists(
            feature_plot
        ):

            st.image(
                feature_plot,
                use_container_width=True
            )

    with xgb_tab2:

        shap_summary = os.path.join(
            PLOT_DIR,
            "xgboost_shap_summary.png"
        )

        shap_bar = os.path.join(
            PLOT_DIR,
            "xgboost_shap_bar.png"
        )

        if os.path.exists(
            shap_summary
        ):

            st.image(
                shap_summary,
                use_container_width=True,
                caption="SHAP Summary"
            )

        if os.path.exists(
            shap_bar
        ):

            st.image(
                shap_bar,
                use_container_width=True,
                caption="SHAP Feature Importance"
            )


# ============================================================
# PAGE 5 — HEALTH ALERTS
# ============================================================

elif page == "🚨 Health Alerts":

    st.markdown(
        '<div class="main-title">'
        'Health Alerts'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Check the pollution category and recommended health action.'
        '</div>',
        unsafe_allow_html=True
    )

    # ----------------------------------------
    # Input
    # ----------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Check PM2.5 Health Alert'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Enter one PM2.5 reading. The category, alert, and recommendation follow the reference table below."
    )

    with st.form("health_alert_form"):
        pm25_value = st.number_input(
            "PM2.5 concentration (µg/m³)",
            min_value=0.0,
            max_value=1000.0,
            value=30.0,
            step=1.0
        )
        alert_submitted = st.form_submit_button(
            "🚨 Check Health Alert",
            type="primary",
            use_container_width=True
        )

    if alert_submitted:
        category = get_aqi_category(pm25_value)

        alert = (
            generate_alert(
                category
            )
        )

        recommendation = (
            get_health_recommendation(
                category
            )
        )

        # ------------------------------------
        # Results
        # ------------------------------------

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "PM2.5",
                f"{pm25_value:.2f}"
            )

        with col2:

            st.metric(
                "Air Quality",
                category
            )

        with col3:

            st.metric(
                "Alert Level",
                alert
            )

        st.divider()

        st.markdown(
            '<div class="section-title">'
            'Recommended Action'
            '</div>',
            unsafe_allow_html=True
        )

        if alert == "NORMAL":

            st.success(
                recommendation
            )

        elif alert in [
            "LOW",
            "MODERATE"
        ]:

            st.warning(
                recommendation
            )

        else:

            st.error(
                recommendation
            )

    # ----------------------------------------
    # Reference table
    # ----------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Air Quality Reference'
        '</div>',
        unsafe_allow_html=True
    )

    reference_data = pd.DataFrame({

        "PM2.5 Range": [
            "0 – 30",
            "31 – 60",
            "61 – 90",
            "91 – 120",
            "121 – 250",
            "250+"
        ],

        "Category": [
            "Good",
            "Satisfactory",
            "Moderately Polluted",
            "Poor",
            "Very Poor",
            "Severe"
        ],

        "Alert": [
            "NORMAL",
            "LOW",
            "MODERATE",
            "HIGH",
            "VERY HIGH",
            "CRITICAL"
        ]
    })

    st.dataframe(
        reference_data,
        use_container_width=True,
        hide_index=True
    )

    # ----------------------------------------
    # Week 7 test results
    # ----------------------------------------

    if alert_results is not None:

        with st.expander(
            "View Week 7 Alert Test Results"
        ):

            st.dataframe(
                alert_results,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# END
# ============================================================