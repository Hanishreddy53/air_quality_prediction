# ============================================================
# WEEK 4 - FEATURE ENGINEERING AND DATA PREPARATION
# Air Quality Prediction System
# ============================================================

import pandas as pd
import numpy as np
import os
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# PART 1 - CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("outputs/plots/feature_engineering", exist_ok=True)
os.makedirs("models", exist_ok=True)


print("=" * 70)
print("WEEK 4 - FEATURE ENGINEERING")
print("=" * 70)


# ============================================================
# PART 2 - LOAD CLEANED DATASET
# ============================================================

data_path = "data/processed/cleaned_dataset.csv"

df = pd.read_csv(data_path)

print("\nCleaned dataset loaded successfully.")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# PART 3 - CONVERT DATETIME
# ============================================================

df["datetime"] = pd.to_datetime(df["datetime"])

# Sort by station and datetime.
# This is very important before creating lag and rolling features.

df = df.sort_values(
    by=["station", "datetime"]
).reset_index(drop=True)

print("\nData sorted station-wise and chronologically.")


# ============================================================
# PART 4 - TIME-BASED FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("TIME FEATURE ENGINEERING")
print("=" * 70)


# Year
df["Year"] = df["datetime"].dt.year

# Month
df["Month"] = df["datetime"].dt.month

# Day
df["Day"] = df["datetime"].dt.day

# Hour
df["Hour"] = df["datetime"].dt.hour

# Day of week
# Monday = 0
# Sunday = 6

df["DayOfWeek"] = df["datetime"].dt.dayofweek

# Weekend feature
# 1 = Weekend
# 0 = Weekday

df["Weekend"] = (
    df["DayOfWeek"] >= 5
).astype(int)


# ============================================================
# PART 5 - SEASON FEATURE
# ============================================================

def get_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    elif month in [3, 4, 5]:
        return "Spring"

    elif month in [6, 7, 8]:
        return "Summer"

    else:
        return "Autumn"


df["Season"] = df["Month"].apply(get_season)

print("\nSeason distribution:")
print(df["Season"].value_counts())


# ============================================================
# PART 6 - RUSH HOUR FEATURE
# ============================================================

def get_rush_hour(hour):

    if hour in [7, 8, 9, 17, 18, 19, 20]:
        return 1

    return 0


df["RushHour"] = df["Hour"].apply(get_rush_hour)

print("\nRush Hour distribution:")
print(df["RushHour"].value_counts())


# ============================================================
# PART 7 - CYCLICAL TIME FEATURES
# ============================================================
#
# Hour 23 and Hour 0 are actually close to each other.
# Normal numerical representation does not show this.
#
# sin/cos representation captures the cyclic nature of time.
# ============================================================

df["Hour_sin"] = np.sin(
    2 * np.pi * df["Hour"] / 24
)

df["Hour_cos"] = np.cos(
    2 * np.pi * df["Hour"] / 24
)

df["Month_sin"] = np.sin(
    2 * np.pi * df["Month"] / 12
)

df["Month_cos"] = np.cos(
    2 * np.pi * df["Month"] / 12
)

print("\nCyclical time features created.")


# ============================================================
# PART 8 - LAG FEATURES
# ============================================================
#
# Lag means using previous observations.
#
# Example:
#
# PM2.5_lag_1
# = PM2.5 value from previous hour
#
# PM2.5_lag_24
# = PM2.5 value from previous day
#
# IMPORTANT:
# Lag features are created separately for each station.
# ============================================================

print("\n" + "=" * 70)
print("CREATING LAG FEATURES")
print("=" * 70)


lag_values = [1, 3, 6, 12, 24]

for lag in lag_values:

    df[f"PM2.5_lag_{lag}"] = (
        df.groupby("station")["PM2.5"]
        .shift(lag)
    )


print("\nLag features created:")
print([
    f"PM2.5_lag_{lag}"
    for lag in lag_values
])


# ============================================================
# PART 9 - POLLUTANT LAG FEATURES
# ============================================================
#
# We also create selected lag features for other pollutants.
# This helps the model understand previous pollution conditions.
# ============================================================

pollutant_lag_columns = [
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3"
]

for column in pollutant_lag_columns:

    df[f"{column}_lag_1"] = (
        df.groupby("station")[column]
        .shift(1)
    )


print("\nSelected pollutant lag features created.")


# ============================================================
# PART 10 - ROLLING AVERAGE FEATURES
# ============================================================
#
# Rolling averages smooth short-term fluctuations.
#
# We use shift(1) BEFORE rolling.
#
# Therefore the current PM2.5 value is NOT included in its
# own rolling average.
#
# This helps avoid target leakage.
# ============================================================

print("\n" + "=" * 70)
print("CREATING ROLLING FEATURES")
print("=" * 70)


rolling_windows = [3, 6, 24]

for window in rolling_windows:

    df[f"PM2.5_rolling_mean_{window}h"] = (
        df.groupby("station")["PM2.5"]
        .transform(
            lambda x:
            x.shift(1)
             .rolling(
                 window=window,
                 min_periods=1
             )
             .mean()
        )
    )


print("\nRolling average features created.")


# ============================================================
# PART 11 - ROLLING STANDARD DEVIATION
# ============================================================
#
# This measures how much pollution fluctuated during the
# previous time window.
# ============================================================

df["PM2.5_rolling_std_24h"] = (
    df.groupby("station")["PM2.5"]
    .transform(
        lambda x:
        x.shift(1)
         .rolling(
             window=24,
             min_periods=2
         )
         .std()
    )
)


print("24-hour rolling standard deviation created.")


# ============================================================
# PART 12 - INTERACTION FEATURES
# ============================================================
#
# Interaction features allow the model to capture relationships
# between different variables.
# ============================================================


# PM10 to PM2.5 relationship
df["PM10_PM2.5_ratio"] = (
    df["PM10"] /
    (df["PM2.5"] + 1)
)


# NO2 and CO interaction
df["NO2_CO_interaction"] = (
    df["NO2"] * df["CO"]
)


# Temperature and Dew Point interaction
df["TEMP_DEWP_interaction"] = (
    df["TEMP"] * df["DEWP"]
)


# Wind speed and PM2.5 interaction
df["WSPM_PM2.5_interaction"] = (
    df["WSPM"] * df["PM2.5"]
)


# Rain and PM2.5 interaction
df["RAIN_PM2.5_interaction"] = (
    df["RAIN"] * df["PM2.5"]
)


print("\nInteraction features created.")


# ============================================================
# PART 13 - AQI CALCULATION
# ============================================================
#
# We calculate pollutant sub-indices using CPCB-style
# breakpoint interpolation.
#
# CPCB AQI has six categories:
#
# 0-50       Good
# 51-100     Satisfactory
# 101-200    Moderately Polluted
# 201-300    Poor
# 301-400    Very Poor
# 401-500    Severe
#
# CPCB states that the worst pollutant sub-index determines
# the overall AQI.
#
# NOTE:
# The Beijing dataset does not contain NH3 and Pb.
# Therefore we calculate AQI using the available pollutants:
# PM10, PM2.5, NO2, SO2, CO and O3.
# ============================================================


# ------------------------------------------------------------
# CPCB breakpoint tables
# ------------------------------------------------------------

aqi_breakpoints = {

    "PM2.5": [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500)
    ],

    "PM10": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 600, 401, 500)
    ],

    "NO2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 800, 401, 500)
    ],

    "SO2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 2000, 401, 500)
    ],

    "CO": [
        (0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10.0, 101, 200),
        (10.1, 17.0, 201, 300),
        (17.1, 34.0, 301, 400),
        (34.1, 50.0, 401, 500)
    ],

    "O3": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 1000, 401, 500)
    ]
}


# ------------------------------------------------------------
# AQI sub-index function
# ------------------------------------------------------------

def calculate_sub_index(concentration, breakpoints):

    if pd.isna(concentration):
        return np.nan

    for c_low, c_high, i_low, i_high in breakpoints:

        if c_low <= concentration <= c_high:

            sub_index = (
                (
                    i_high - i_low
                ) /
                (
                    c_high - c_low
                )
            ) * (
                concentration - c_low
            ) + i_low

            return sub_index

    # Values above the final breakpoint
    if concentration > breakpoints[-1][1]:

        return 500

    return np.nan


# ============================================================
# PART 14 - CO UNIT CONVERSION
# ============================================================
#
# The Beijing dataset reports CO in micrograms/m3.
#
# CPCB CO breakpoints are expressed in mg/m3.
#
# 1 mg/m3 = 1000 micrograms/m3
#
# Therefore:
#
# CO_mg_m3 = CO / 1000
# ============================================================

df["CO_mg_m3"] = df["CO"] / 1000


# ============================================================
# PART 15 - CALCULATE INDIVIDUAL AQI SUB-INDICES
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING AQI SUB-INDICES")
print("=" * 70)


# PM2.5
df["AQI_PM2.5"] = df["PM2.5"].apply(
    lambda x:
    calculate_sub_index(
        x,
        aqi_breakpoints["PM2.5"]
    )
)


# PM10
df["AQI_PM10"] = df["PM10"].apply(
    lambda x:
    calculate_sub_index(
        x,
        aqi_breakpoints["PM10"]
    )
)


# NO2
df["AQI_NO2"] = df["NO2"].apply(
    lambda x:
    calculate_sub_index(
        x,
        aqi_breakpoints["NO2"]
    )
)


# SO2
df["AQI_SO2"] = df["SO2"].apply(
    lambda x:
    calculate_sub_index(
        x,
        aqi_breakpoints["SO2"]
    )
)


# CO
df["AQI_CO"] = df["CO_mg_m3"].apply(
    lambda x:
    calculate_sub_index(
        x,
        aqi_breakpoints["CO"]
    )
)


# O3
df["AQI_O3"] = df["O3"].apply(
    lambda x:
    calculate_sub_index(
        x,
        aqi_breakpoints["O3"]
    )
)


# ============================================================
# PART 16 - OVERALL AQI
# ============================================================
#
# CPCB method:
# The worst pollutant sub-index determines overall AQI.
# ============================================================

aqi_columns = [
    "AQI_PM2.5",
    "AQI_PM10",
    "AQI_NO2",
    "AQI_SO2",
    "AQI_CO",
    "AQI_O3"
]

df["AQI"] = df[aqi_columns].max(axis=1)


# Keep AQI within 0-500
df["AQI"] = df["AQI"].clip(
    lower=0,
    upper=500
)


print("\nAQI calculated successfully.")


# ============================================================
# PART 17 - AQI CATEGORY
# ============================================================

def get_aqi_category(aqi):

    if pd.isna(aqi):
        return np.nan

    elif aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderately Polluted"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


df["AQI_Category"] = df["AQI"].apply(
    get_aqi_category
)


print("\nAQI Category Distribution:")
print(
    df["AQI_Category"].value_counts()
)


# ============================================================
# PART 18 - AQI CATEGORY ENCODING
# ============================================================

label_encoder = LabelEncoder()

df["AQI_Class"] = label_encoder.fit_transform(
    df["AQI_Category"]
)


print("\nAQI Category Mapping:")

for category, encoded_value in zip(
    label_encoder.classes_,
    label_encoder.transform(
        label_encoder.classes_
    )
):

    print(
        category,
        "->",
        encoded_value
    )


# Save label encoder
joblib.dump(
    label_encoder,
    "models/aqi_label_encoder.pkl"
)


# ============================================================
# PART 19 - REMOVE UNNECESSARY COLUMNS
# ============================================================
#
# No is only an identifier.
#
# datetime has already been converted into useful features.
#
# CO_mg_m3 is only an intermediate AQI calculation column.
# ============================================================

columns_to_remove = [
    "No",
    "datetime",
    "CO_mg_m3"
]

df.drop(
    columns=columns_to_remove,
    errors="ignore",
    inplace=True
)


# ============================================================
# PART 20 - HANDLE NEW MISSING VALUES
# ============================================================
#
# Lag and rolling features naturally create missing values
# at the beginning of each station's time series.
#
# We use backward fill for these engineered features.
#
# IMPORTANT:
# We do NOT use the target value to fill missing target values.
# ============================================================

engineered_columns = []

for lag in lag_values:

    engineered_columns.append(
        f"PM2.5_lag_{lag}"
    )

for column in pollutant_lag_columns:

    engineered_columns.append(
        f"{column}_lag_1"
    )

for window in rolling_windows:

    engineered_columns.append(
        f"PM2.5_rolling_mean_{window}h"
    )

engineered_columns.append(
    "PM2.5_rolling_std_24h"
)


for column in engineered_columns:

    if column in df.columns:

        df[column] = (
            df.groupby("station")[column]
            .transform(
                lambda x: x.bfill().ffill()
            )
        )


# ============================================================
# PART 21 - HANDLE ANY REMAINING MISSING VALUES
# ============================================================
#
# This is a safety check after feature engineering.
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns


for column in numeric_columns:

    if df[column].isnull().sum() > 0:

        df[column] = df[column].fillna(
            df[column].median()
        )


categorical_columns = df.select_dtypes(
    include="object"
).columns


for column in categorical_columns:

    if df[column].isnull().sum() > 0:

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


print("\nRemaining missing values:")
print(df.isnull().sum().sum())


# ============================================================
# PART 22 - SAVE FEATURE-ENGINEERED DATASET
# ============================================================

df.to_csv(
    "data/processed/feature_engineered_dataset.csv",
    index=False
)

print(
    "\nFeature-engineered dataset saved successfully."
)


# ============================================================
# PART 23 - ENCODE CATEGORICAL FEATURES
# ============================================================
#
# station, wind direction and season are categorical variables.
#
# One-hot encoding converts them into numerical columns.
# ============================================================

categorical_features = [
    "station",
    "wd",
    "Season"
]


df_encoded = pd.get_dummies(
    df,
    columns=categorical_features,
    drop_first=True
)


# Convert Boolean columns into integers
boolean_columns = df_encoded.select_dtypes(
    include="bool"
).columns


df_encoded[boolean_columns] = (
    df_encoded[boolean_columns]
    .astype(int)
)


print("\nCategorical encoding completed.")

print(
    "Shape after encoding:",
    df_encoded.shape
)


# ============================================================
# PART 24 - SAVE ENCODED DATASET
# ============================================================

df_encoded.to_csv(
    "data/processed/encoded_dataset.csv",
    index=False
)


# ============================================================
# PART 25 - REGRESSION FEATURES AND TARGET
# ============================================================
#
# Regression objective:
#
# Predict future/current PM2.5 concentration.
#
# PM2.5 is the target and therefore must not be present
# inside X.
# ============================================================

regression_target = "PM2.5"


X_reg = df_encoded.drop(
    columns=[
        "PM2.5",
        "AQI",
        "AQI_Category",
        "AQI_Class"
    ],
    errors="ignore"
)

y_reg = df_encoded[
    regression_target
]


# ============================================================
# PART 26 - CLASSIFICATION FEATURES AND TARGET
# ============================================================
#
# Classification objective:
#
# Predict AQI category.
#
# We exclude current AQI and its component sub-indices.
#
# We also exclude PM2.5 because AQI is strongly dependent on
# PM2.5 and using the current PM2.5 value as an input would
# make the classification task artificially easy.
#
# Historical PM2.5 lag features are retained because they
# represent previous observations.
# ============================================================

aqi_columns_to_remove = [
    "AQI",
    "AQI_Category",
    "AQI_Class",
    "AQI_PM2.5",
    "AQI_PM10",
    "AQI_NO2",
    "AQI_SO2",
    "AQI_CO",
    "AQI_O3",
    "PM2.5"
]


X_class = df_encoded.drop(
    columns=aqi_columns_to_remove,
    errors="ignore"
)

y_class = df_encoded[
    "AQI_Class"
]


print("\nRegression dataset:")
print("X:", X_reg.shape)
print("y:", y_reg.shape)


print("\nClassification dataset:")
print("X:", X_class.shape)
print("y:", y_class.shape)


# ============================================================
# PART 27 - CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================
#
# Because this is time-series data, we do not randomly shuffle
# the observations.
#
# First 80% -> Training
# Last 20%  -> Testing
# ============================================================

split_index = int(
    len(df_encoded) * 0.80
)


# Regression split

X_reg_train = X_reg.iloc[
    :split_index
]

X_reg_test = X_reg.iloc[
    split_index:
]

y_reg_train = y_reg.iloc[
    :split_index
]

y_reg_test = y_reg.iloc[
    split_index:
]


# Classification split

X_class_train = X_class.iloc[
    :split_index
]

X_class_test = X_class.iloc[
    split_index:
]

y_class_train = y_class.iloc[
    :split_index
]

y_class_test = y_class.iloc[
    split_index:
]


print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

print("\nRegression:")
print("Training:", X_reg_train.shape)
print("Testing :", X_reg_test.shape)


print("\nClassification:")
print("Training:", X_class_train.shape)
print("Testing :", X_class_test.shape)


# ============================================================
# PART 28 - FEATURE SCALING
# ============================================================
#
# StandardScaler standardizes numerical features.
#
# IMPORTANT:
# fit_transform() is applied ONLY to training data.
#
# Testing data uses transform().
#
# This prevents data leakage.
# ============================================================

reg_scaler = StandardScaler()

X_reg_train_scaled = (
    reg_scaler.fit_transform(
        X_reg_train
    )
)

X_reg_test_scaled = (
    reg_scaler.transform(
        X_reg_test
    )
)


class_scaler = StandardScaler()

X_class_train_scaled = (
    class_scaler.fit_transform(
        X_class_train
    )
)

X_class_test_scaled = (
    class_scaler.transform(
        X_class_test
    )
)


print("\nFeature scaling completed.")


# ============================================================
# PART 29 - SAVE SCALERS
# ============================================================

joblib.dump(
    reg_scaler,
    "models/regression_scaler.pkl"
)

joblib.dump(
    class_scaler,
    "models/classification_scaler.pkl"
)

print("\nScalers saved successfully.")


# ============================================================
# PART 30 - SAVE FEATURE NAMES
# ============================================================

regression_features = X_reg.columns.tolist()

classification_features = X_class.columns.tolist()


pd.DataFrame({
    "Feature": regression_features
}).to_csv(
    "outputs/reports/regression_features.csv",
    index=False
)


pd.DataFrame({
    "Feature": classification_features
}).to_csv(
    "outputs/reports/classification_features.csv",
    index=False
)


# ============================================================
# PART 31 - SAVE TRAINING AND TESTING DATA
# ============================================================
#
# Saving these datasets makes Week 5 model training easier.
# ============================================================


# Regression training data

reg_train = X_reg_train.copy()

reg_train["PM2.5"] = y_reg_train.values


reg_train.to_csv(
    "data/processed/regression_train.csv",
    index=False
)


# Regression testing data

reg_test = X_reg_test.copy()

reg_test["PM2.5"] = y_reg_test.values


reg_test.to_csv(
    "data/processed/regression_test.csv",
    index=False
)


# Classification training data

class_train = X_class_train.copy()

class_train["AQI_Class"] = (
    y_class_train.values
)


class_train.to_csv(
    "data/processed/classification_train.csv",
    index=False
)


# Classification testing data

class_test = X_class_test.copy()

class_test["AQI_Class"] = (
    y_class_test.values
)


class_test.to_csv(
    "data/processed/classification_test.csv",
    index=False
)


# ============================================================
# PART 32 - FEATURE ENGINEERING SUMMARY
# ============================================================

feature_summary = pd.DataFrame({

    "Feature_Group": [
        "Time Features",
        "Cyclical Features",
        "Lag Features",
        "Rolling Features",
        "Interaction Features",
        "AQI Features",
        "Categorical Features"
    ],

    "Description": [
        "Year, Month, Day, Hour, DayOfWeek, Weekend, RushHour",
        "Hour sine/cosine and Month sine/cosine",
        "Previous PM2.5 and pollutant observations",
        "3h, 6h, 24h rolling averages and 24h rolling std",
        "Pollutant and weather interaction features",
        "Pollutant sub-indices, AQI and AQI category",
        "Station, wind direction and season encoding"
    ]
})


feature_summary.to_csv(
    "outputs/reports/week4_feature_summary.csv",
    index=False
)


# ============================================================
# PART 33 - FINAL DATASET SUMMARY
# ============================================================

dataset_summary = pd.DataFrame({

    "Dataset": [
        "Original Cleaned Dataset",
        "Feature Engineered Dataset",
        "Encoded Dataset",
        "Regression Training",
        "Regression Testing",
        "Classification Training",
        "Classification Testing"
    ],

    "Rows": [
        len(pd.read_csv(data_path)),
        len(df),
        len(df_encoded),
        len(X_reg_train),
        len(X_reg_test),
        len(X_class_train),
        len(X_class_test)
    ],

    "Columns": [
        len(pd.read_csv(data_path).columns),
        len(df.columns),
        len(df_encoded.columns),
        len(regression_features) + 1,
        len(regression_features) + 1,
        len(classification_features) + 1,
        len(classification_features) + 1
    ]
})


dataset_summary.to_csv(
    "outputs/reports/week4_dataset_summary.csv",
    index=False
)


# ============================================================
# PART 34 - FINAL CHECKS
# ============================================================

print("\n" + "=" * 70)
print("FINAL WEEK 4 CHECKS")
print("=" * 70)


print("\nFeature Engineered Dataset:")
print(df.shape)


print("\nEncoded Dataset:")
print(df_encoded.shape)


print("\nTotal Missing Values:")
print(df_encoded.isnull().sum().sum())


print("\nTotal Duplicate Rows:")
print(df_encoded.duplicated().sum())


print("\nAQI Statistics:")
print(df["AQI"].describe())


print("\nAQI Category Distribution:")
print(
    df["AQI_Category"].value_counts()
)


# ============================================================
# PART 35 - COMPLETION MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("WEEK 4 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated important files:")

print("1. feature_engineered_dataset.csv")
print("2. encoded_dataset.csv")
print("3. regression_train.csv")
print("4. regression_test.csv")
print("5. classification_train.csv")
print("6. classification_test.csv")

print("\nGenerated preprocessing objects:")

print("7. regression_scaler.pkl")
print("8. classification_scaler.pkl")
print("9. aqi_label_encoder.pkl")

print("\nGenerated reports:")

print("10. regression_features.csv")
print("11. classification_features.csv")
print("12. week4_feature_summary.csv")
print("13. week4_dataset_summary.csv")

print("\nReady for Week 5 - Machine Learning Model Training.")