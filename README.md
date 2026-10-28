# Air Quality Prediction System

An end-to-end air-quality analytics and machine-learning project built from the Beijing Multi-Site Air-Quality dataset. It combines measurements from 12 monitoring stations, cleans and analyzes pollutant/weather data, engineers temporal and AQI features, trains regression and classification models, produces explainability reports, and serves predictions and health alerts through Streamlit.

## What this project does

- Combines the 12 station CSV files in `data/raw/` into one dataset.
- Builds a station-aware `datetime` column from year, month, day, and hour.
- Reports and repairs missing pollutant, weather, and wind-direction values.
- Explores distributions, correlations, outliers, station differences, seasonal behavior, and time-series patterns.
- Creates temporal, cyclical, lag, rolling-window, interaction, and AQI features.
- Trains models for two machine-learning objectives:
  - **Regression:** predict numerical PM2.5 concentration.
  - **Classification:** predict an AQI category.
- Compares baseline models and optimizes a Random Forest regressor and XGBoost classifier.
- Generates feature importance and optional SHAP explanations.
- Provides a Streamlit interface for predictions, AQI categories, alert levels, and health recommendations.

## End-to-end workflow

Run the stages from the project root in this order. Each stage reads files created by the previous stage.

| Stage | File | What it does | Main outputs |
| --- | --- | --- | --- |
| Week 2 | `week2_dataset_understanding.py` | Reads every raw station CSV, adds a station identifier, concatenates the files, inspects shape/types/duplicates/ranges, and summarizes pollutants, weather, stations, and dates. | `data/processed/combined_dataset.csv`, initial dataset reports |
| Week 3 | `week3_data_cleaning_eda.py` | Creates `datetime`; interpolates and forward/backward fills pollutant/weather values station-wise; fills wind direction with the station mode; validates data; analyzes univariate, bivariate, correlation, outlier, station, season, and time-series patterns. | `cleaned_dataset.csv`, missing-value reports, EDA reports and plots |
| Week 4 | `week4_feature_engineering.py` | Adds time, season, rush-hour, cyclical, lag, rolling, interaction, AQI, encoded, and train/test features. | `feature_engineered_dataset.csv`, `encoded_dataset.csv`, train/test files, scalers, feature reports |
| Week 5 | `week5_model_training.py` | Trains and evaluates four regression models and five classification models, selects the best models, saves comparisons, confusion matrices, feature importance, and classification metrics. | Best baseline models, comparison reports, model-result plots |
| Week 6 | `week6_model_optimization.py` | Prepares leakage-controlled regression/classification data, runs randomized hyperparameter searches, compares baseline versus optimized models, and saves optimized models and importance plots. | Optimized model files, optimization reports, confusion matrix, importance reports |
| Week 7 | `week7_explainable_ai_alerts.py` | Reloads optimized models, recreates their feature schema, saves feature importance and SHAP plots, tests six alert examples, and writes a completion summary. | Week 7 reports and plots |
| Week 8 | `app.py` and `week8_app.py` | Launches the Streamlit dashboard using the processed data, optimized models, and Week 7 outputs. | Interactive dashboard |

There is no separate Week 1 script in this repository; Week 2 is the first included implementation stage.

## Data

### Raw input

The `data/raw/` directory contains one CSV per monitoring station:

`Aotizhongxin`, `Changping`, `Dingling`, `Dongsi`, `Guanyuan`, `Gucheng`, `Huairou`, `Nongzhanguan`, `Shunyi`, `Tiantan`, `Wanliu`, and `Wanshouxigong`.

The source measurements include:

- Pollutants: `PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, and `O3`
- Weather: `TEMP`, `PRES`, `DEWP`, `RAIN`, and `WSPM`
- Wind direction: `wd`
- Time fields: `year`, `month`, `day`, and `hour`

### Cleaning behavior

- Pollutant and weather columns are interpolated within each station, then forward-filled and backward-filled.
- Missing `wd` values are replaced with the station's most common wind direction.
- A missing-value report is saved before cleaning and a comparison is saved after cleaning.
- Cleaning is performed station-wise so values do not cross station boundaries.

## Feature engineering

`week4_feature_engineering.py` sorts records by station and time before creating features.

### Time features

- `Year`, `Month`, `Day`, `Hour`, and `DayOfWeek`
- `Weekend`
- `Season`: Winter, Spring, Summer, or Autumn
- `RushHour`: hours 7–9 and 17–20
- Sine/cosine encodings for hour and month

### Historical and rolling features

- PM2.5 lags: 1, 3, 6, 12, and 24 hours
- One-hour lags for `PM10`, `NO2`, `SO2`, `CO`, and `O3`
- PM2.5 rolling means over 3, 6, and 24 hours
- PM2.5 rolling standard deviation over 24 hours

Rolling values use the previous observations (`shift(1)`) to avoid including the current target value and causing target leakage. Initial engineered missing values are backfilled.

### Interaction features

- `PM10_PM2.5_ratio`
- `NO2_CO_interaction`
- `TEMP_DEWP_interaction`
- `WSPM_PM2.5_interaction`
- `RAIN_PM2.5_interaction`

### AQI calculation

AQI sub-indices are calculated using CPCB-style breakpoint interpolation for PM2.5, PM10, NO2, SO2, CO, and O3. CO is converted from µg/m³ to mg/m³ before applying the CO breakpoints. The overall AQI is the maximum pollutant sub-index and is clipped to 0–500.

| AQI range | Category |
| --- | --- |
| 0–50 | Good |
| 51–100 | Satisfactory |
| 101–200 | Moderately Polluted |
| 201–300 | Poor |
| 301–400 | Very Poor |
| 401–500 | Severe |

The dataset does not include NH3 or Pb, so those pollutants are not used in the AQI calculation. `AQI_Category` is label-encoded as `AQI_Class` and the encoder is saved for reuse.

## Machine-learning design

### Regression

The target is `PM2.5`. Week 5 compares:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Regression metrics are MAE, RMSE, and R². Week 6 optimizes a Random Forest Regressor with `RandomizedSearchCV` and saves the optimized model as `optimized_random_forest_regressor.pkl`.

### Classification

The target is the AQI category/class. Current AQI and its component sub-indices are excluded from the Week 6 classification features to reduce direct target leakage; historical PM2.5 lag features remain available.

Week 5 compares:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Decision Tree Classifier
- XGBoost Classifier

Classification metrics include accuracy, weighted F1, and a classification report. Week 6 optimizes an XGBoost Classifier and saves `optimized_xgboost_classifier.pkl`.

### Preprocessing and saved models

Categorical columns are one-hot encoded with `drop_first=True`, boolean columns are converted to integers, infinite values are replaced, and missing numeric values are handled before model training. Scalers and the AQI label encoder are saved under `models/` for reproducibility.

## Dashboard

Start the application with:

```text
streamlit run app.py
```

`app.py` is the entry point and imports `week8_app.py`. The dashboard uses cached data/model loading and expects these files:

- `data/processed/feature_engineered_dataset.csv`
- `models/optimized_random_forest_regressor.pkl`
- `models/optimized_xgboost_classifier.pkl`
- `outputs/reports/regression_feature_importance.csv`
- `outputs/reports/classification_feature_importance.csv`
- `outputs/reports/week7_prediction_alert_results.csv`
- Week 7 plots under `outputs/plots/week7/`

### Home

Displays the project overview, record count, feature count, station count, average PM2.5, and model availability.

### Prediction

Accepts:

- PM2.5, PM10, SO2, NO2, CO, and O3
- Temperature, pressure, dew point, rain, and wind speed
- Hour, month, and season

The form accepts CO in mg/m³ and converts it to the dataset's µg/m³ convention for model input. It predicts PM2.5 with the optimized Random Forest, calculates the input AQI, displays the AQI category and alert level, and provides a health recommendation.

### Health Alerts

Accepts a PM2.5 value and returns its category, alert level, and recommendation:

| Category | Alert |
| --- | --- |
| Good | NORMAL |
| Satisfactory | LOW |
| Moderately Polluted | MODERATE |
| Poor | HIGH |
| Very Poor | VERY HIGH |
| Severe | CRITICAL |

The recommendations are project guidance for demonstration, not medical advice.

### Explainability and analytics code

`week7_explainable_ai_alerts.py` generates feature-importance and SHAP assets from the saved models. `week8_app.py` contains additional analytics and explainability rendering code for saved statistics, feature importance, and SHAP images. The visible sidebar navigation should be checked against the current `week8_app.py` version if those optional dashboard sections are enabled or changed.

## Setup

Use Python 3.10 or newer. From the project root on Windows:

```text
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib streamlit shap
```

`shap` is optional: Week 7 still generates the other reports when SHAP is unavailable, but SHAP plots are skipped. Run all commands from the repository root because the scripts use relative paths.

## Running the complete pipeline

```text
python week2_dataset_understanding.py
python week3_data_cleaning_eda.py
python week4_feature_engineering.py
python week5_model_training.py
python week6_model_optimization.py
python week7_explainable_ai_alerts.py
streamlit run app.py
```

The scripts create required directories automatically. Re-running a stage can overwrite its generated datasets, reports, plots, or model files.

## Repository structure

```text
.
├── app.py
├── week2_dataset_understanding.py
├── week3_data_cleaning_eda.py
├── week4_feature_engineering.py
├── week5_model_training.py
├── week6_model_optimization.py
├── week7_explainable_ai_alerts.py
├── week8_app.py
├── data/
│   ├── raw/                 # 12 original station CSV files
│   └── processed/           # generated datasets and train/test splits
├── models/                  # trained models, scalers, and encoders
└── outputs/
    ├── plots/               # generated visualizations
    └── reports/             # generated CSV, TXT, and PNG reports
```

## Generated datasets

| File | Description |
| --- | --- |
| `combined_dataset.csv` | Concatenated raw station data with a station column. |
| `cleaned_dataset.csv` | Dataset after missing-value handling. |
| `feature_engineered_dataset.csv` | Dataset with temporal, lag, rolling, interaction, and AQI features. |
| `encoded_dataset.csv` | Encoded feature-engineered dataset. |
| `regression_train.csv`, `regression_test.csv` | Regression splits with PM2.5 as target. |
| `classification_train.csv`, `classification_test.csv` | Classification splits with `AQI_Class` as target. |

## Saved models and preprocessing artifacts

| File | Purpose |
| --- | --- |
| `best_regression_model.pkl` | Best Week 5 baseline regression model. |
| `best_classification_model.pkl` | Best Week 5 baseline classification model. |
| `optimized_random_forest_regressor.pkl` | Week 6 optimized PM2.5 regressor used by the app. |
| `optimized_xgboost_classifier.pkl` | Week 6 optimized AQI classifier used by the app. |
| `regression_scaler.pkl` | Regression feature scaler. |
| `classification_scaler.pkl` | Classification feature scaler. |
| `aqi_label_encoder.pkl` | Encoder for AQI category labels. |

## Reports

Generated reports are stored in `outputs/reports/` and include:

- **Data quality:** `missing_value_report.csv`, `missing_value_comparison.csv`, `outlier_report.csv`
- **EDA:** `univariate_summary.csv`, `correlation_matrix.csv`, `pm25_correlation.csv`, `pm25_feature_correlation.csv`, `station_report.csv`, `season_report.csv`, `hourly_pm25.csv`, `monthly_pm25.csv`, `yearly_pm25.csv`
- **Feature engineering:** `aqi_feature_correlation_week4.csv`, `week4_dataset_summary.csv`, `week4_feature_summary.csv`, `selected_features.csv`
- **Feature lists and splits:** `regression_features.csv`, `classification_features.csv`
- **Baseline models:** `regression_model_comparison.csv`, `classification_model_comparison.csv`, `best_classification_report.txt`, `best_model_summary.csv`, `regression_feature_importance.csv`, `classification_feature_importance.csv`
- **Baseline importance images:** `regression_feature_importance.png`, `classification_feature_importance.png`
- **Optimization:** `random_forest_optimization.csv`, `xgboost_optimization.csv`, `optimized_random_forest_feature_importance.csv`, `optimized_xgboost_feature_importance.csv`
- **Alerts:** `alert_system_test_results.csv`, `week7_prediction_alert_results.csv`, `week7_summary.csv`

## Plot groups

The `outputs/plots/` directory contains:

- `univariate/` — pollutant and weather histograms and boxplots
- `bivariate/` — PM2.5 relationships, regression plots, and correlation plots
- `correlation/` — correlation heatmap and selected-feature plot
- `outliers/` — pollutant/weather boxplots
- `station_season/` — station and seasonal pollutant comparisons
- `time_series/` — hourly, monthly, and yearly pollution trends
- `feature_engineering/` — AQI distribution and AQI feature correlation
- `model_results/` — baseline metric comparisons, predictions, and confusion matrix
- `model_optimization/` — optimized feature importance and XGBoost confusion matrix
- `week7/` — alert distribution, feature importance, and SHAP summary/bar plots

## Important limitations

- The scripts depend on the existing raw CSV schema and relative paths.
- AQI is calculated from the available pollutants only; NH3 and Pb are absent from the source data.
- The AQI thresholds and health recommendations are for project demonstration and should not replace official local air-quality information or professional medical guidance.
- `joblib` model files are executable serialized artifacts and should only be loaded from trusted sources.
- Model results depend on the installed versions of Python libraries and the generated train/test data.
- The project currently has no `requirements.txt`; install the packages listed above or create a pinned environment for reproducible deployment.
