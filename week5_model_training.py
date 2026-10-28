# ============================================================
# WEEK 5 - MACHINE LEARNING MODEL TRAINING
# Air Quality Prediction Project
# ============================================================

import pandas as pd
import numpy as np
import os
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# MACHINE LEARNING IMPORTS
# ============================================================

# Regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor


# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# Evaluation metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,

    accuracy_score,
    precision_score,
    recall_score,
    f1_score,

    classification_report,
    confusion_matrix
)


# ============================================================
# PART 1 - CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs("models", exist_ok=True)

os.makedirs(
    "outputs/plots/model_results",
    exist_ok=True
)

os.makedirs(
    "outputs/reports",
    exist_ok=True
)


print("=" * 70)
print("WEEK 5 - MACHINE LEARNING MODEL TRAINING")
print("=" * 70)


# ============================================================
# PART 2 - LOAD REGRESSION DATA
# ============================================================

reg_train = pd.read_csv(
    "data/processed/regression_train.csv"
)

reg_test = pd.read_csv(
    "data/processed/regression_test.csv"
)


print("\nRegression training data:")
print(reg_train.shape)

print("\nRegression testing data:")
print(reg_test.shape)


# ============================================================
# PART 3 - PREPARE REGRESSION DATA
# ============================================================

X_reg_train = reg_train.drop(
    columns=["PM2.5"]
)

y_reg_train = reg_train["PM2.5"]


X_reg_test = reg_test.drop(
    columns=["PM2.5"]
)

y_reg_test = reg_test["PM2.5"]


print("\nRegression features:")
print(X_reg_train.shape)

print("Regression target:")
print(y_reg_train.shape)


# ============================================================
# PART 4 - TRAIN REGRESSION MODELS
# ============================================================

print("\n")
print("=" * 70)
print("REGRESSION MODEL TRAINING")
print("=" * 70)


regression_models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),

    "XGBoost":
        XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror"
        )
}


regression_results = []

trained_regression_models = {}


# ============================================================
# PART 5 - TRAIN AND EVALUATE REGRESSION MODELS
# ============================================================

for name, model in regression_models.items():

    print("\nTraining:", name)

    # Train model
    model.fit(
        X_reg_train,
        y_reg_train
    )

    # Prediction
    predictions = model.predict(
        X_reg_test
    )

    # Metrics
    mae = mean_absolute_error(
        y_reg_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_reg_test,
            predictions
        )
    )

    r2 = r2_score(
        y_reg_test,
        predictions
    )

    regression_results.append({

        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_regression_models[name] = model

    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2  :", round(r2, 4))


# ============================================================
# PART 6 - REGRESSION RESULTS TABLE
# ============================================================

regression_results_df = pd.DataFrame(
    regression_results
)

regression_results_df = (
    regression_results_df
    .sort_values(
        by="RMSE"
    )
)


print("\n")
print("=" * 70)
print("REGRESSION MODEL COMPARISON")
print("=" * 70)

print(
    regression_results_df
)


# Save regression results

regression_results_df.to_csv(
    "outputs/reports/regression_model_comparison.csv",
    index=False
)


# ============================================================
# PART 7 - FIND BEST REGRESSION MODEL
# ============================================================
#
# Lower RMSE = better
# Higher R2 = better
# ============================================================

best_regression_name = (
    regression_results_df
    .iloc[0]["Model"]
)

best_regression_model = (
    trained_regression_models[
        best_regression_name
    ]
)


print(
    "\nBest Regression Model:",
    best_regression_name
)


# Save best regression model

joblib.dump(
    best_regression_model,
    "models/best_regression_model.pkl"
)


# ============================================================
# PART 8 - REGRESSION MODEL COMPARISON PLOT
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=regression_results_df,
    x="Model",
    y="RMSE"
)

plt.title(
    "Regression Model Comparison - RMSE"
)

plt.xlabel("Model")
plt.ylabel("RMSE")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "outputs/plots/model_results/regression_rmse_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# PART 9 - R2 COMPARISON PLOT
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=regression_results_df,
    x="Model",
    y="R2 Score"
)

plt.title(
    "Regression Model Comparison - R2 Score"
)

plt.xlabel("Model")
plt.ylabel("R2 Score")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "outputs/plots/model_results/regression_r2_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# PART 10 - ACTUAL VS PREDICTED
# ============================================================

best_regression_predictions = (
    best_regression_model.predict(
        X_reg_test
    )
)


plt.figure(figsize=(8, 6))

plt.scatter(
    y_reg_test,
    best_regression_predictions,
    alpha=0.3
)

plt.xlabel(
    "Actual PM2.5"
)

plt.ylabel(
    "Predicted PM2.5"
)

plt.title(
    "Actual vs Predicted PM2.5"
)

# Perfect prediction line

min_value = min(
    y_reg_test.min(),
    best_regression_predictions.min()
)

max_value = max(
    y_reg_test.max(),
    best_regression_predictions.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value]
)

plt.tight_layout()

plt.savefig(
    "outputs/plots/model_results/actual_vs_predicted_pm25.png",
    dpi=300
)

plt.close()


# ============================================================
# PART 11 - LOAD CLASSIFICATION DATA
# ============================================================

class_train = pd.read_csv(
    "data/processed/classification_train.csv"
)

class_test = pd.read_csv(
    "data/processed/classification_test.csv"
)


print("\n")
print("=" * 70)
print("CLASSIFICATION DATA")
print("=" * 70)

print(
    "Training:",
    class_train.shape
)

print(
    "Testing:",
    class_test.shape
)


# ============================================================
# PART 12 - PREPARE CLASSIFICATION DATA
# ============================================================

X_class_train = class_train.drop(
    columns=["AQI_Class"]
)

y_class_train = class_train[
    "AQI_Class"
]


X_class_test = class_test.drop(
    columns=["AQI_Class"]
)

y_class_test = class_test[
    "AQI_Class"
]


print(
    "\nClassification features:",
    X_class_train.shape
)

print(
    "Classification target:",
    y_class_train.shape
)


# ============================================================
# PART 13 - TRAIN CLASSIFICATION MODELS
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION MODEL TRAINING")
print("=" * 70)


classification_models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42,
            max_depth=10
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss"
        )
}


classification_results = []

trained_classification_models = {}


# ============================================================
# PART 14 - TRAIN AND EVALUATE CLASSIFICATION MODELS
# ============================================================

for name, model in classification_models.items():

    print("\nTraining:", name)

    # Train
    model.fit(
        X_class_train,
        y_class_train
    )

    # Prediction
    predictions = model.predict(
        X_class_test
    )

    # Metrics
    accuracy = accuracy_score(
        y_class_test,
        predictions
    )

    precision = precision_score(
        y_class_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_class_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_class_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    classification_results.append({

        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_classification_models[name] = model

    print(
        "Accuracy :",
        round(accuracy, 4)
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall   :",
        round(recall, 4)
    )

    print(
        "F1 Score :",
        round(f1, 4)
    )


# ============================================================
# PART 15 - CLASSIFICATION RESULTS TABLE
# ============================================================

classification_results_df = pd.DataFrame(
    classification_results
)

classification_results_df = (
    classification_results_df
    .sort_values(
        by="F1 Score",
        ascending=False
    )
)


print("\n")
print("=" * 70)
print("CLASSIFICATION MODEL COMPARISON")
print("=" * 70)

print(
    classification_results_df
)


classification_results_df.to_csv(
    "outputs/reports/classification_model_comparison.csv",
    index=False
)


# ============================================================
# PART 16 - FIND BEST CLASSIFICATION MODEL
# ============================================================

best_classification_name = (
    classification_results_df
    .iloc[0]["Model"]
)

best_classification_model = (
    trained_classification_models[
        best_classification_name
    ]
)


print(
    "\nBest Classification Model:",
    best_classification_name
)


# Save best model

joblib.dump(
    best_classification_model,
    "models/best_classification_model.pkl"
)


# ============================================================
# PART 17 - CLASSIFICATION COMPARISON PLOT
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=classification_results_df,
    x="Model",
    y="F1 Score"
)

plt.title(
    "Classification Model Comparison - F1 Score"
)

plt.xlabel("Model")
plt.ylabel("F1 Score")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "outputs/plots/model_results/classification_f1_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# PART 18 - ACCURACY COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=classification_results_df,
    x="Model",
    y="Accuracy"
)

plt.title(
    "Classification Model Comparison - Accuracy"
)

plt.xlabel("Model")
plt.ylabel("Accuracy")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "outputs/plots/model_results/classification_accuracy_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# PART 19 - CONFUSION MATRIX
# ============================================================

best_class_predictions = (
    best_classification_model.predict(
        X_class_test
    )
)


cm = confusion_matrix(
    y_class_test,
    best_class_predictions
)


plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Confusion Matrix - Best Classification Model"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.tight_layout()

plt.savefig(
    "outputs/plots/model_results/best_classification_confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# PART 20 - CLASSIFICATION REPORT
# ============================================================

classification_report_text = (
    classification_report(
        y_class_test,
        best_class_predictions,
        zero_division=0
    )
)


print("\n")
print("=" * 70)
print("BEST CLASSIFICATION MODEL REPORT")
print("=" * 70)

print(
    classification_report_text
)


with open(
    "outputs/reports/best_classification_report.txt",
    "w"
) as file:

    file.write(
        classification_report_text
    )


# ============================================================
# PART 21 - FEATURE IMPORTANCE
# ============================================================
#
# Tree-based models provide feature importance.
#
# We use the best tree-based model if possible.
# ============================================================


def save_feature_importance(
    model,
    feature_names,
    filename,
    title
):

    if hasattr(
        model,
        "feature_importances_"
    ):

        importance = (
            model.feature_importances_
        )

        feature_importance = pd.DataFrame({

            "Feature": feature_names,

            "Importance": importance
        })

        feature_importance = (
            feature_importance
            .sort_values(
                by="Importance",
                ascending=False
            )
        )

        feature_importance.to_csv(
            filename,
            index=False
        )

        # Top 20 features

        top_features = (
            feature_importance
            .head(20)
        )

        plt.figure(
            figsize=(10, 8)
        )

        sns.barplot(
            data=top_features,
            y="Feature",
            x="Importance"
        )

        plt.title(title)

        plt.tight_layout()

        plt.savefig(
            filename.replace(
                ".csv",
                ".png"
            ),
            dpi=300
        )

        plt.close()

        return feature_importance

    else:

        print(
            "\nFeature importance is not directly available "
            "for this model."
        )

        return None


# Regression feature importance

regression_importance = (
    save_feature_importance(

        best_regression_model,

        X_reg_train.columns,

        "outputs/reports/regression_feature_importance.csv",

        "Top Regression Features"
    )
)


# Classification feature importance

classification_importance = (
    save_feature_importance(

        best_classification_model,

        X_class_train.columns,

        "outputs/reports/classification_feature_importance.csv",

        "Top Classification Features"
    )
)


# ============================================================
# PART 22 - FINAL MODEL SUMMARY
# ============================================================

model_summary = pd.DataFrame({

    "Task": [
        "Regression",
        "Classification"
    ],

    "Objective": [
        "Predict PM2.5 concentration",
        "Predict AQI category"
    ],

    "Best Model": [
        best_regression_name,
        best_classification_name
    ]
})


model_summary.to_csv(
    "outputs/reports/best_model_summary.csv",
    index=False
)


print("\n")
print("=" * 70)
print("BEST MODELS")
print("=" * 70)

print(
    model_summary
)


# ============================================================
# PART 23 - FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("WEEK 5 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nRegression models trained:",
    len(regression_models)
)

print(
    "Classification models trained:",
    len(classification_models)
)

print(
    "\nBest Regression Model:",
    best_regression_name
)

print(
    "Best Classification Model:",
    best_classification_name
)

print(
    "\nModels saved in: models/"
)

print(
    "\nReports saved in: outputs/reports/"
)

print(
    "\nPlots saved in: outputs/plots/model_results/"
)

print(
    "\nReady for Week 6."
)