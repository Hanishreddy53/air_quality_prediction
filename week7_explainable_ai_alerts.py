"""Week 7: explainability, AQI mapping, and health alerts."""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    shap = None

warnings.filterwarnings("ignore")
ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "feature_engineered_dataset.csv"
REGRESSION_MODEL_PATH = ROOT_DIR / "models" / "optimized_random_forest_regressor.pkl"
CLASSIFICATION_MODEL_PATH = ROOT_DIR / "models" / "optimized_xgboost_classifier.pkl"
REPORT_DIR = ROOT_DIR / "outputs" / "reports"
PLOT_DIR = ROOT_DIR / "outputs" / "plots" / "week7"


def prepare_features(data: pd.DataFrame, model) -> pd.DataFrame:
    """Reproduce the Week 6 preprocessing and align to the saved model schema."""
    prepared = data.copy()
    if "PM2.5" in prepared.columns:
        prepared = prepared.drop(columns=["PM2.5"])
    for column in ("datetime", "No", "station"):
        if column in prepared.columns:
            prepared = prepared.drop(columns=[column])
    categorical = prepared.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical:
        prepared = pd.get_dummies(prepared, columns=categorical, drop_first=True, dtype=np.int8)
    boolean_columns = prepared.select_dtypes(include=["bool"]).columns.tolist()
    if boolean_columns:
        prepared[boolean_columns] = prepared[boolean_columns].astype(np.int8)
    prepared = prepared.replace([np.inf, -np.inf], np.nan).fillna(0)
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        expected = [str(column) for column in expected]
        for column in expected:
            if column not in prepared.columns:
                prepared[column] = 0
        prepared = prepared[expected]
    return prepared


def get_aqi_category(pm25: float) -> str:
    value = float(pm25)
    if value <= 30:
        return "Good"
    if value <= 60:
        return "Satisfactory"
    if value <= 90:
        return "Moderately Polluted"
    if value <= 120:
        return "Poor"
    if value <= 250:
        return "Very Poor"
    return "Severe"


def generate_alert(category: str) -> str:
    return {"Good": "NORMAL", "Satisfactory": "LOW", "Moderately Polluted": "MODERATE", "Poor": "HIGH", "Very Poor": "VERY HIGH", "Severe": "CRITICAL"}.get(category, "UNKNOWN")


def get_health_recommendation(category: str) -> str:
    return {
        "Good": "The air is clean and comfortable. Outdoor exercise and normal daily activities are safe.",
        "Satisfactory": "Conditions are generally acceptable. People with asthma should take short breaks outdoors and monitor symptoms.",
        "Moderately Polluted": "Limit long outdoor exercise, especially for children and older adults. Choose indoor activities when possible.",
        "Poor": "Reduce outdoor exposure and keep windows closed during peak pollution. Sensitive groups should use a well-fitted mask outdoors.",
        "Very Poor": "Stay indoors as much as possible and avoid strenuous activity. Use air filtration and follow local health updates.",
        "Severe": "Treat this as an emergency air-quality condition. Remain indoors, avoid all outdoor exertion, and seek medical help for breathing problems.",
    }.get(category, "No recommendation available.")


def _save_shap_plots(model, features: pd.DataFrame, prefix: str) -> str:
    if shap is None:
        return "Skipped (SHAP is not installed)"
    try:
        sample = features.sample(min(1000, len(features)), random_state=42)
        values = shap.TreeExplainer(model).shap_values(sample)
        if isinstance(values, list):
            values = values[0]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 0]
        for plot_type, filename in ((None, f"{prefix}_shap_summary.png"), ("bar", f"{prefix}_shap_bar.png")):
            plt.figure()
            shap.summary_plot(values, sample, plot_type=plot_type, show=False)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / filename, dpi=200, bbox_inches="tight")
            plt.close()
        return "Completed"
    except Exception as error:
        plt.close("all")
        return f"Skipped ({type(error).__name__}: {error})"


def run_week7() -> dict[str, str]:
    """Generate Week 7 reports, plots, SHAP explanations, and alert tests."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("WEEK 7 - EXPLAINABLE AI AND ALERT SYSTEM")
    print("=" * 70)
    df = pd.read_csv(DATA_PATH)
    rf_model = joblib.load(REGRESSION_MODEL_PATH)
    xgb_model = joblib.load(CLASSIFICATION_MODEL_PATH)
    X_rf = prepare_features(df, rf_model)
    X_xgb = prepare_features(df, xgb_model)
    statuses: dict[str, str] = {}

    for name, model, features, filename, plot_name, title in (
        ("Random Forest Feature Importance", rf_model, X_rf, "regression_feature_importance.csv", "random_forest_feature_importance.png", "Random Forest - Top Feature Importance"),
        ("XGBoost Feature Importance", xgb_model, X_xgb, "classification_feature_importance.csv", "xgboost_feature_importance.png", "XGBoost - Top Feature Importance"),
    ):
        importance = pd.DataFrame({"Feature": features.columns, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        importance.to_csv(REPORT_DIR / filename, index=False)
        top = importance.head(15).iloc[::-1]
        plt.figure(figsize=(10, 7))
        plt.barh(top["Feature"], top["Importance"])
        plt.title(title)
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / plot_name, dpi=200, bbox_inches="tight")
        plt.close()
        statuses[name] = "Completed"

    statuses["Random Forest SHAP"] = _save_shap_plots(rf_model, X_rf, "random_forest")
    statuses["XGBoost SHAP"] = _save_shap_plots(xgb_model, X_xgb, "xgboost")
    test_rows = []
    for pm25 in (20, 45, 75, 105, 180, 300):
        category = get_aqi_category(pm25)
        test_rows.append({"PM2.5": pm25, "AQI_Category": category, "Alert_Level": generate_alert(category), "Health_Recommendation": get_health_recommendation(category)})
    pd.DataFrame(test_rows).to_csv(REPORT_DIR / "week7_prediction_alert_results.csv", index=False)
    statuses.update({"AQI Category Mapping": "Completed", "Health Recommendation": "Completed", "Intelligent Alert System": "Completed"})
    pd.DataFrame({"Component": list(statuses), "Status": list(statuses.values())}).to_csv(REPORT_DIR / "week7_summary.csv", index=False)
    print("Week 7 completed. Reports and plots saved under outputs/.")
    return statuses


if __name__ == "__main__":
    run_week7()
