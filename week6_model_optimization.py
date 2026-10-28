# ============================================================
# WEEK 6 - MODEL OPTIMIZATION
# Air Quality Prediction Project
# ============================================================


# ============================================================
# PART 1 - IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import os
import warnings
import joblib

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# PART 2 - CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs("models", exist_ok=True)

os.makedirs("outputs/reports", exist_ok=True)

os.makedirs(
    "outputs/plots/model_optimization",
    exist_ok=True
)

print("=" * 70)
print("WEEK 6 - MODEL OPTIMIZATION")
print("=" * 70)


# ============================================================
# PART 3 - LOAD WEEK 4 FEATURE ENGINEERED DATASET
# ============================================================

print("\nPART 3 - LOADING FEATURE ENGINEERED DATASET")
print("-" * 70)

df = pd.read_csv(
    "data/processed/feature_engineered_dataset.csv"
)

print("Dataset loaded successfully!")

print("Original Shape:", df.shape)


# ============================================================
# PART 4 - CHECK DATA
# ============================================================

print("\nPART 4 - CHECKING DATA")
print("-" * 70)

print("Columns:")

print(df.columns.tolist())

print("\nData Types:")

print(df.dtypes)


# ============================================================
# PART 5 - REMOVE UNNECESSARY COLUMNS
# ============================================================

print("\nPART 5 - REMOVING UNNECESSARY COLUMNS")
print("-" * 70)


# datetime is already represented by
# hour/day/month/year features.

if "datetime" in df.columns:

    df.drop(
        columns=["datetime"],
        inplace=True
    )


# No is just the original row number.
# It is not useful for prediction.

if "No" in df.columns:

    df.drop(
        columns=["No"],
        inplace=True
    )


# Station is a station identifier.
# We remove it because our model should learn
# environmental relationships rather than the ID.

if "station" in df.columns:

    df.drop(
        columns=["station"],
        inplace=True
    )


print("Unnecessary columns removed.")


# ============================================================
# PART 6 - HANDLE CATEGORICAL FEATURES
# ============================================================

print("\nPART 6 - HANDLING CATEGORICAL FEATURES")
print("-" * 70)


# Find text/categorical columns.

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()


print("Categorical columns found:")

print(categorical_columns)


# Convert categorical columns to numerical columns.

if len(categorical_columns) > 0:

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True,
        dtype=np.int8
    )


print("\nCategorical encoding completed.")


# Check again.

remaining_categorical = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()


print(
    "Remaining categorical columns:",
    remaining_categorical
)


# This prevents the exact "Spring" error.

if len(remaining_categorical) > 0:

    raise ValueError(
        "Categorical columns still exist: "
        + str(remaining_categorical)
    )


print("\nAll features are numerical.")


# ============================================================
# PART 7 - HANDLE INFINITE AND MISSING VALUES
# ============================================================

print("\nPART 7 - HANDLING MISSING VALUES")
print("-" * 70)


# Replace infinity values with NaN.

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# Find numerical columns.

numeric_columns = df.select_dtypes(
    include=np.number
).columns


# Fill missing values using median.

df[numeric_columns] = df[
    numeric_columns
].fillna(
    df[numeric_columns].median()
)


print(
    "Remaining missing values:",
    df.isnull().sum().sum()
)


print("Missing values handled successfully.")


# ============================================================
# PART 8 - REGRESSION DATA
# ============================================================

print("\nPART 8 - PREPARING REGRESSION DATA")
print("-" * 70)


regression_target = "PM2.5"


if regression_target not in df.columns:

    raise ValueError(
        "PM2.5 column not found."
    )


# Features

X_reg = df.drop(
    columns=[regression_target]
)


# Target

y_reg = df[
    regression_target
]


print(
    "Regression Target:",
    regression_target
)

print(
    "Number of Features:",
    X_reg.shape[1]
)

print(
    "Number of Rows:",
    X_reg.shape[0]
)


# ============================================================
# PART 9 - FINAL SAFETY CHECK
# ============================================================

print("\nPART 9 - FINAL FEATURE CHECK")
print("-" * 70)


text_columns = X_reg.select_dtypes(
    include=["object", "category"]
).columns.tolist()


print(
    "Text columns found:",
    text_columns
)


if len(text_columns) > 0:

    raise ValueError(
        "STOP! Text columns still exist: "
        + str(text_columns)
    )


print(
    "All regression features are numerical."
)


# ============================================================
# PART 10 - REGRESSION TRAIN TEST SPLIT
# ============================================================

print("\nPART 10 - REGRESSION TRAIN TEST SPLIT")
print("-" * 70)


X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(

    X_reg,

    y_reg,

    test_size=0.20,

    random_state=42
)


print(
    "Training Rows:",
    len(X_reg_train)
)

print(
    "Testing Rows:",
    len(X_reg_test)
)


# ============================================================
# PART 11 - CREATE OPTIMIZATION SAMPLE
# ============================================================

print("\nPART 11 - CREATING OPTIMIZATION SAMPLE")
print("-" * 70)


# Full hyperparameter search on 336K rows
# would take a long time.
#
# Therefore, we use a representative sample
# for finding good hyperparameters.

optimization_size = min(
    20000,
    len(X_reg_train)
)


X_reg_opt, _, y_reg_opt, _ = train_test_split(

    X_reg_train,

    y_reg_train,

    train_size=optimization_size,

    random_state=42
)


print(
    "Optimization rows:",
    len(X_reg_opt)
)


# ============================================================
# PART 12 - RANDOM FOREST BASELINE
# ============================================================

print("\nPART 12 - RANDOM FOREST BASELINE")
print("-" * 70)


rf_baseline = RandomForestRegressor(

    n_estimators=50,

    random_state=42,

    n_jobs=1
)


print(
    "Training baseline Random Forest..."
)


rf_baseline.fit(

    X_reg_train,

    y_reg_train
)


print(
    "Baseline Random Forest training completed."
)


# Prediction

rf_baseline_pred = rf_baseline.predict(
    X_reg_test
)


# Metrics

rf_baseline_mae = mean_absolute_error(

    y_reg_test,

    rf_baseline_pred
)


rf_baseline_rmse = np.sqrt(

    mean_squared_error(

        y_reg_test,

        rf_baseline_pred
    )
)


rf_baseline_r2 = r2_score(

    y_reg_test,

    rf_baseline_pred
)


print("\nBaseline Random Forest Results")

print(
    "MAE :",
    rf_baseline_mae
)

print(
    "RMSE:",
    rf_baseline_rmse
)

print(
    "R2  :",
    rf_baseline_r2
)


# ============================================================
# PART 13 - RANDOM FOREST HYPERPARAMETER OPTIMIZATION
# ============================================================

print("\nPART 13 - RANDOM FOREST OPTIMIZATION")
print("-" * 70)


rf_model = RandomForestRegressor(

    random_state=42,

    n_jobs=1
)


# Small search space to reduce execution time.

rf_param_grid = {

    "n_estimators": [50, 100],

    "max_depth": [10, 20],

    "min_samples_split": [2, 5]

}


rf_random_search = RandomizedSearchCV(

    estimator=rf_model,

    param_distributions=rf_param_grid,

    n_iter=3,

    cv=2,

    scoring="neg_root_mean_squared_error",

    random_state=42,

    n_jobs=1,

    verbose=1
)


print(
    "Starting Random Forest optimization..."
)


rf_random_search.fit(

    X_reg_opt,

    y_reg_opt
)


print(
    "\nBest Random Forest Parameters:"
)

print(
    rf_random_search.best_params_
)


print(
    "\nBest Cross Validation RMSE:"
)

print(
    -rf_random_search.best_score_
)


# ============================================================
# PART 14 - OPTIMIZED RANDOM FOREST
# ============================================================

print("\nPART 14 - EVALUATING OPTIMIZED RANDOM FOREST")
print("-" * 70)


best_rf = rf_random_search.best_estimator_


rf_optimized_pred = best_rf.predict(

    X_reg_test
)


rf_optimized_mae = mean_absolute_error(

    y_reg_test,

    rf_optimized_pred
)


rf_optimized_rmse = np.sqrt(

    mean_squared_error(

        y_reg_test,

        rf_optimized_pred
    )
)


rf_optimized_r2 = r2_score(

    y_reg_test,

    rf_optimized_pred
)


print("\nOptimized Random Forest Results")

print(
    "MAE :",
    rf_optimized_mae
)

print(
    "RMSE:",
    rf_optimized_rmse
)

print(
    "R2  :",
    rf_optimized_r2
)


# ============================================================
# PART 15 - RANDOM FOREST COMPARISON
# ============================================================

print("\nPART 15 - RANDOM FOREST COMPARISON")
print("-" * 70)


rf_comparison = pd.DataFrame({

    "Model": [

        "Random Forest - Week 5",

        "Random Forest - Optimized"

    ],

    "MAE": [

        rf_baseline_mae,

        rf_optimized_mae

    ],

    "RMSE": [

        rf_baseline_rmse,

        rf_optimized_rmse

    ],

    "R2 Score": [

        rf_baseline_r2,

        rf_optimized_r2

    ]

})


print(rf_comparison)


rf_comparison.to_csv(

    "outputs/reports/random_forest_optimization.csv",

    index=False
)


# ============================================================
# PART 16 - CREATE AQI CATEGORY
# ============================================================

print("\nPART 16 - CREATING AQI CATEGORY")
print("-" * 70)


def create_aqi_category(pm25):

    if pm25 <= 50:

        return "Good"

    elif pm25 <= 100:

        return "Moderate"

    elif pm25 <= 150:

        return "Unhealthy for Sensitive Groups"

    elif pm25 <= 200:

        return "Unhealthy"

    elif pm25 <= 300:

        return "Very Unhealthy"

    else:

        return "Hazardous"


df["AQI_Category"] = df[
    "PM2.5"
].apply(
    create_aqi_category
)


print(
    df["AQI_Category"].value_counts()
)


# ============================================================
# PART 17 - ENCODE AQI CATEGORY
# ============================================================

print("\nPART 17 - ENCODING AQI CATEGORY")
print("-" * 70)


category_mapping = {

    "Good": 0,

    "Moderate": 1,

    "Unhealthy for Sensitive Groups": 2,

    "Unhealthy": 3,

    "Very Unhealthy": 4,

    "Hazardous": 5

}


df["AQI_Category_Code"] = df[
    "AQI_Category"
].map(
    category_mapping
)


print(
    "AQI categories converted to numerical labels."
)


# ============================================================
# PART 18 - CLASSIFICATION DATA
# ============================================================

print("\nPART 18 - PREPARING CLASSIFICATION DATA")
print("-" * 70)


# IMPORTANT:
#
# PM2.5 is NOT included as a feature here because
# AQI_Category was created directly from PM2.5.
#
# Including PM2.5 would cause data leakage.

X_class = df.drop(

    columns=[
        "PM2.5",
        "AQI_Category",
        "AQI_Category_Code"
    ]
)


y_class = df[
    "AQI_Category_Code"
]


print(
    "Classification Target: AQI_Category"
)

print(
    "Number of Features:",
    X_class.shape[1]
)


# ============================================================
# PART 19 - CLASSIFICATION TRAIN TEST SPLIT
# ============================================================

print("\nPART 19 - CLASSIFICATION TRAIN TEST SPLIT")
print("-" * 70)


X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(

    X_class,

    y_class,

    test_size=0.20,

    random_state=42,

    stratify=y_class
)


print(
    "Training Rows:",
    len(X_class_train)
)

print(
    "Testing Rows:",
    len(X_class_test)
)


# ============================================================
# PART 20 - CLASSIFICATION OPTIMIZATION SAMPLE
# ============================================================

print("\nPART 20 - CLASSIFICATION OPTIMIZATION SAMPLE")
print("-" * 70)


classification_optimization_size = min(

    20000,

    len(X_class_train)
)


X_class_opt, _, y_class_opt, _ = train_test_split(

    X_class_train,

    y_class_train,

    train_size=classification_optimization_size,

    random_state=42,

    stratify=y_class_train
)


print(
    "Optimization rows:",
    len(X_class_opt)
)


# ============================================================
# PART 21 - XGBOOST BASELINE
# ============================================================

print("\nPART 21 - XGBOOST BASELINE")
print("-" * 70)


xgb_baseline = XGBClassifier(

    n_estimators=100,

    max_depth=6,

    learning_rate=0.1,

    random_state=42,

    n_jobs=1,

    eval_metric="mlogloss"
)


print(
    "Training baseline XGBoost..."
)


xgb_baseline.fit(

    X_class_train,

    y_class_train
)


print(
    "Baseline XGBoost training completed."
)


xgb_baseline_pred = xgb_baseline.predict(

    X_class_test
)


# Metrics

xgb_baseline_accuracy = accuracy_score(

    y_class_test,

    xgb_baseline_pred
)


xgb_baseline_precision = precision_score(

    y_class_test,

    xgb_baseline_pred,

    average="weighted",

    zero_division=0
)


xgb_baseline_recall = recall_score(

    y_class_test,

    xgb_baseline_pred,

    average="weighted",

    zero_division=0
)


xgb_baseline_f1 = f1_score(

    y_class_test,

    xgb_baseline_pred,

    average="weighted",

    zero_division=0
)


print("\nBaseline XGBoost Results")

print(
    "Accuracy :",
    xgb_baseline_accuracy
)

print(
    "Precision:",
    xgb_baseline_precision
)

print(
    "Recall   :",
    xgb_baseline_recall
)

print(
    "F1 Score :",
    xgb_baseline_f1
)


# ============================================================
# PART 22 - XGBOOST OPTIMIZATION
# ============================================================

print("\nPART 22 - XGBOOST OPTIMIZATION")
print("-" * 70)


xgb_model = XGBClassifier(

    random_state=42,

    n_jobs=1,

    eval_metric="mlogloss"
)


xgb_param_grid = {

    "n_estimators": [50, 100],

    "max_depth": [3, 5],

    "learning_rate": [0.05, 0.1]

}


xgb_random_search = RandomizedSearchCV(

    estimator=xgb_model,

    param_distributions=xgb_param_grid,

    n_iter=3,

    cv=2,

    scoring="f1_weighted",

    random_state=42,

    n_jobs=1,

    verbose=1
)


print(
    "Starting XGBoost optimization..."
)


xgb_random_search.fit(

    X_class_opt,

    y_class_opt
)


print(
    "\nBest XGBoost Parameters:"
)

print(
    xgb_random_search.best_params_
)


print(
    "\nBest Cross Validation F1 Score:"
)

print(
    xgb_random_search.best_score_
)


# ============================================================
# PART 23 - OPTIMIZED XGBOOST
# ============================================================

print("\nPART 23 - EVALUATING OPTIMIZED XGBOOST")
print("-" * 70)


best_xgb = xgb_random_search.best_estimator_


xgb_optimized_pred = best_xgb.predict(

    X_class_test
)


xgb_optimized_accuracy = accuracy_score(

    y_class_test,

    xgb_optimized_pred
)


xgb_optimized_precision = precision_score(

    y_class_test,

    xgb_optimized_pred,

    average="weighted",

    zero_division=0
)


xgb_optimized_recall = recall_score(

    y_class_test,

    xgb_optimized_pred,

    average="weighted",

    zero_division=0
)


xgb_optimized_f1 = f1_score(

    y_class_test,

    xgb_optimized_pred,

    average="weighted",

    zero_division=0
)


print("\nOptimized XGBoost Results")

print(
    "Accuracy :",
    xgb_optimized_accuracy
)

print(
    "Precision:",
    xgb_optimized_precision
)

print(
    "Recall   :",
    xgb_optimized_recall
)

print(
    "F1 Score :",
    xgb_optimized_f1
)


# ============================================================
# PART 24 - XGBOOST COMPARISON
# ============================================================

print("\nPART 24 - XGBOOST COMPARISON")
print("-" * 70)


xgb_comparison = pd.DataFrame({

    "Model": [

        "XGBoost - Week 5",

        "XGBoost - Optimized"

    ],

    "Accuracy": [

        xgb_baseline_accuracy,

        xgb_optimized_accuracy

    ],

    "Precision": [

        xgb_baseline_precision,

        xgb_optimized_precision

    ],

    "Recall": [

        xgb_baseline_recall,

        xgb_optimized_recall

    ],

    "F1 Score": [

        xgb_baseline_f1,

        xgb_optimized_f1

    ]

})


print(xgb_comparison)


xgb_comparison.to_csv(

    "outputs/reports/xgboost_optimization.csv",

    index=False
)


# ============================================================
# PART 25 - RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\nPART 25 - RANDOM FOREST FEATURE IMPORTANCE")
print("-" * 70)


rf_importance = pd.DataFrame({

    "Feature":
    X_reg.columns,

    "Importance":
    best_rf.feature_importances_

})


rf_importance.sort_values(

    by="Importance",

    ascending=False,

    inplace=True
)


print(
    "\nTop 15 Random Forest Features:"
)

print(
    rf_importance.head(15)
)


rf_importance.to_csv(

    "outputs/reports/optimized_random_forest_feature_importance.csv",

    index=False
)


# Plot

top_rf = rf_importance.head(15)


plt.figure(
    figsize=(10, 6)
)


plt.barh(

    top_rf["Feature"][::-1],

    top_rf["Importance"][::-1]
)


plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 15 Features - Optimized Random Forest"
)


plt.tight_layout()


plt.savefig(

    "outputs/plots/model_optimization/optimized_random_forest_feature_importance.png"
)


plt.close()


print(
    "Random Forest feature importance plot saved."
)


# ============================================================
# PART 26 - XGBOOST FEATURE IMPORTANCE
# ============================================================

print("\nPART 26 - XGBOOST FEATURE IMPORTANCE")
print("-" * 70)


xgb_importance = pd.DataFrame({

    "Feature":
    X_class.columns,

    "Importance":
    best_xgb.feature_importances_

})


xgb_importance.sort_values(

    by="Importance",

    ascending=False,

    inplace=True
)


print(
    "\nTop 15 XGBoost Features:"
)

print(
    xgb_importance.head(15)
)


xgb_importance.to_csv(

    "outputs/reports/optimized_xgboost_feature_importance.csv",

    index=False
)


# Plot

top_xgb = xgb_importance.head(15)


plt.figure(
    figsize=(10, 6)
)


plt.barh(

    top_xgb["Feature"][::-1],

    top_xgb["Importance"][::-1]
)


plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 15 Features - Optimized XGBoost"
)


plt.tight_layout()


plt.savefig(

    "outputs/plots/model_optimization/optimized_xgboost_feature_importance.png"
)


plt.close()


print(
    "XGBoost feature importance plot saved."
)


# ============================================================
# PART 27 - CONFUSION MATRIX
# ============================================================

print("\nPART 27 - CONFUSION MATRIX")
print("-" * 70)


cm = confusion_matrix(

    y_class_test,

    xgb_optimized_pred
)


display_labels = [

    "Good",

    "Moderate",

    "USG",

    "Unhealthy",

    "Very Unhealthy",

    "Hazardous"

]


# Only use labels that actually exist
# in the test data.

existing_labels = sorted(
    y_class_test.unique()
)


existing_display_labels = [
    display_labels[i]
    for i in existing_labels
]


disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=existing_display_labels
)


fig, ax = plt.subplots(

    figsize=(8, 6)
)


disp.plot(

    ax=ax,

    xticks_rotation=45,

    colorbar=False
)


plt.title(
    "Optimized XGBoost - Confusion Matrix"
)


plt.tight_layout()


plt.savefig(

    "outputs/plots/model_optimization/optimized_xgboost_confusion_matrix.png"
)


plt.close()


print(
    "Confusion matrix saved."
)


# ============================================================
# PART 28 - SAVE OPTIMIZED MODELS
# ============================================================

print("\nPART 28 - SAVING OPTIMIZED MODELS")
print("-" * 70)


joblib.dump(

    best_rf,

    "models/optimized_random_forest_regressor.pkl"
)


joblib.dump(

    best_xgb,

    "models/optimized_xgboost_classifier.pkl"
)


print(
    "Optimized Random Forest saved."
)

print(
    "Optimized XGBoost saved."
)


# ============================================================
# PART 29 - FINAL SUMMARY
# ============================================================

print("\nPART 29 - FINAL WEEK 6 SUMMARY")
print("=" * 70)


print("\nREGRESSION - RANDOM FOREST")

print(
    "Baseline R2:",
    rf_baseline_r2
)

print(
    "Optimized R2:",
    rf_optimized_r2
)

print(
    "Baseline RMSE:",
    rf_baseline_rmse
)

print(
    "Optimized RMSE:",
    rf_optimized_rmse
)


print("\nCLASSIFICATION - XGBOOST")

print(
    "Baseline Accuracy:",
    xgb_baseline_accuracy
)

print(
    "Optimized Accuracy:",
    xgb_optimized_accuracy
)

print(
    "Baseline F1:",
    xgb_baseline_f1
)

print(
    "Optimized F1:",
    xgb_optimized_f1
)


print("\n" + "=" * 70)

print(
    "WEEK 6 MODEL OPTIMIZATION COMPLETED!"
)

print("=" * 70)