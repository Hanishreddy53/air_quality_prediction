import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

#loading week2 combined dataset
df = pd.read_csv("data/processed/combined_dataset.csv") 

df["datetime"] = pd.to_datetime(
    df[["year","month","day","hour"]]
)

print(df.shape)
print(df.head())
print(df.info())

#part1
#Check Missing Values
missing = df.isnull().sum()
print(missing)

#Missing Value Percentage
missing_percent = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({

    "Column": df.columns,
    "Missing Values": missing.values,
    "Missing Percentage": missing_percent.values

})

missing_df = missing_df.sort_values(
    by="Missing Percentage",
    ascending=False
)

print("\n")
print(missing_df)

#Save Missing Value Report
missing_df.to_csv(

    "outputs/reports/missing_value_report.csv",

    index=False

)
print("\nMissing Value Report Saved")

#Visualize Missing Values
plt.figure(figsize=(12,6))

sns.barplot(

    x="Column",

    y="Missing Percentage",

    data=missing_df

)

plt.xticks(rotation=90)

plt.title("Missing Value Percentage")

plt.tight_layout()

plt.savefig(

    "outputs/plots/missing_value_percentage.png"

)
plt.show()


#part2 is missing value handling
print("Missing Values Before Cleaning")
print(df.isnull().sum())

#Fill Pollutant Values
# Pollutant Columns
# Pollutant Columns
pollutants = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3"
]

print("=" * 60)
print("Handling Missing Values in Pollutant Columns (Station-wise)")
print("=" * 60)

for col in pollutants:

    before = df[col].isnull().sum()

    df[col] = df.groupby("station")[col].transform(
        lambda x: x.interpolate(method="linear")
                  .ffill()
                  .bfill()
    )

    after = df[col].isnull().sum()

    print(f"{col}")
    print(f"Missing Before : {before}")
    print(f"Missing After  : {after}")
    print("-" * 40)

#Fill Weather Columns
weather = [
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM"
]

print("=" * 60)
print("Handling Missing Values in Weather Columns (Station-wise)")
print("=" * 60)

for col in weather:

    before = df[col].isnull().sum()

    df[col] = df.groupby("station")[col].transform(
        lambda x: x.interpolate(method="linear")
                  .ffill()
                  .bfill()
    )

    after = df[col].isnull().sum()

    print(f"{col}")
    print(f"Missing Before : {before}")
    print(f"Missing After  : {after}")
    print("-" * 40)

#Fill Wind Direction
print("=" * 60)
print("Handling Missing Values in Wind Direction")
print("=" * 60)

before = df["wd"].isnull().sum()

df["wd"] = df.groupby("station")["wd"].transform(
    lambda x: x.fillna(x.mode()[0])
)

after = df["wd"].isnull().sum()

print(f"Missing Before : {before}")
print(f"Missing After  : {after}")

#checking missing values again after cleaining
print(df.isnull().sum())

#Save Cleaned Dataset
df.to_csv("data/processed/cleaned_dataset.csv", index=False)
print("Cleaned Dataset Saved")

#Missing Value Comparison
before = missing

after = df.isnull().sum()

comparison = pd.DataFrame({

    "Before Cleaning": before,

    "After Cleaning": after

})

print(comparison)

comparison.to_csv(

    "outputs/reports/missing_value_comparison.csv",

    index=True

)

#Visualize Before vs After
plt.figure(figsize=(12,6))

comparison.plot(
    kind="bar",
    figsize=(12,6)
)

plt.title("Missing Values Before and After Cleaning")

plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "outputs/plots/missing_value_before_after.png"
)

plt.show()

#Data Validation 
print("=" * 60)
print("Data Validation")
print("=" * 60)

print("Duplicate Rows :", df.duplicated().sum())

print("\nTotal Missing Values :", df.isnull().sum().sum())

print("\nDataset Shape :", df.shape)

print("\nNumber of Stations :", df["station"].nunique())

print("\nStation Names :")
print(df["station"].unique())




#part3
print("\n")
print("=" * 60)
print("Week 3 - Part 3 : Outlier Detection")
print("=" * 60)

numerical_columns = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM"
]
os.makedirs("outputs/plots/outliers", exist_ok=True)

#Create Boxplots
for col in numerical_columns:

    plt.figure(figsize=(7,5))

    sns.boxplot(x=df[col])

    plt.title(f"Box Plot of {col}")

    plt.tight_layout()

    plt.savefig(f"outputs/plots/outliers/{col}_boxplot.png")

    plt.close()

print("All Boxplots Saved Successfully")

#Detect Outliers using IQR
outlier_report = []

for col in numerical_columns:

    Q1 = df[col].quantile(0.25)

    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR

    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    outlier_report.append({

        "Column": col,

        "Q1": Q1,

        "Q3": Q3,

        "Lower Limit": lower,

        "Upper Limit": upper,

        "Outlier Count": len(outliers)

    })

#Create Outlier Report
outlier_df = pd.DataFrame(outlier_report)
print(outlier_df)
#Save Report
outlier_df.to_csv(
    "outputs/reports/outlier_report.csv",
    index=False
)
print("Outlier Report Saved Successfully")

#Visualize Outlier Counts
plt.figure(figsize=(12,6))
sns.barplot(
    x="Column",
    y="Outlier Count",
    data=outlier_df
)
plt.xticks(rotation=45)
plt.title("Number of Outliers in Each Feature")
plt.tight_layout()
plt.savefig(
    "outputs/plots/outlier_count.png"
)
plt.show()





#part4  UNIVARIATE ANALYSIS
os.makedirs("outputs/plots/univariate", exist_ok=True)
numerical_columns = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM"
]

#Histogram + KDE
print("=" * 60)
print("Generating Histograms")
print("=" * 60)

for col in numerical_columns:

    plt.figure(figsize=(8,5))

    sns.histplot(
        df[col],
        kde=True,
        bins=30
    )

    plt.title(f"{col} Distribution")

    plt.xlabel(col)

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        f"outputs/plots/univariate/{col}_histogram.png"
    )

    plt.close()

print("Histograms Saved Successfully")

#boxplots
print("=" * 60)
print("Generating Boxplots")
print("=" * 60)

for col in numerical_columns:

    plt.figure(figsize=(8,2))

    sns.boxplot(
        x=df[col]
    )

    plt.title(f"{col} Boxplot")

    plt.tight_layout()

    plt.savefig(
        f"outputs/plots/univariate/{col}_boxplot.png"
    )

    plt.close()

print("Boxplots Saved Successfully")

#Descriptive Statistics
summary = df[numerical_columns].describe().T

summary["Median"] = df[numerical_columns].median()

summary["Variance"] = df[numerical_columns].var()

summary["Skewness"] = df[numerical_columns].skew()

summary["Kurtosis"] = df[numerical_columns].kurt()

#Save Summary
summary.to_csv(
    "outputs/reports/univariate_summary.csv"
)

print("Summary Report Saved")
print(summary)


#Distribution Interpretation
print("=" * 60)
print("Distribution Analysis")
print("=" * 60)

for col in numerical_columns:

    skew = df[col].skew()

    if skew > 1:
        result = "Highly Positively Skewed"

    elif skew > 0.5:
        result = "Moderately Positively Skewed"

    elif skew < -1:
        result = "Highly Negatively Skewed"

    elif skew < -0.5:
        result = "Moderately Negatively Skewed"

    else:
        result = "Approximately Symmetric"

    print(f"{col:<8} : {result}")


#Bivariate Analysis
target = "PM2.5"

features = [
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM"
]
os.makedirs("outputs/plots/bivariate", exist_ok=True)
sample_df = df.sample(10000, random_state=42)

#Scatter Plots
print("="*60)
print("Generating Scatter Plots")
print("="*60)

for col in features:

    plt.figure(figsize=(7,5))

    sns.scatterplot(
        x=df[col],
        y=df[target],
        alpha=0.4
    )

    plt.title(f"{target} vs {col}")

    plt.tight_layout()

    plt.savefig(
        f"outputs/plots/bivariate/{target}_vs_{col}_scatter.png"
    )

    plt.close()

print("Scatter Plots Saved Successfully")

#Regression Plots
print("="*60)
print("Generating Regression Plots")
print("="*60)

for col in features:

    plt.figure(figsize=(7,5))

    sns.regplot(
        x=df[col],
        y=df[target],
        scatter_kws={"alpha":0.3},
        line_kws={"color":"red"}
    )

    plt.title(f"{target} vs {col}")

    plt.tight_layout()

    plt.savefig(
        f"outputs/plots/bivariate/{target}_vs_{col}_regression.png"
    )

    plt.close()

print("Regression Plots Saved Successfully")

#Correlation with PM2.5
correlation = []

for col in features:

    corr = df[target].corr(df[col])

    correlation.append([col, corr])

correlation_df = pd.DataFrame(
    correlation,
    columns=["Feature","Correlation"]
)

print(correlation_df)

#Sort Correlation
correlation_df = correlation_df.sort_values(
    by="Correlation",
    ascending=False
)

print(correlation_df)
correlation_df.to_csv(
    "outputs/reports/pm25_feature_correlation.csv",
    index=False
)

print("Correlation Report Saved")

#Correlation Bar Chart
plt.figure(figsize=(10,6))

sns.barplot(
    x="Correlation",
    y="Feature",
    data=correlation_df
)

plt.title("Correlation with PM2.5")

plt.tight_layout()

plt.savefig(
    "outputs/plots/bivariate/pm25_correlation.png"
)

plt.show()

#Print Strong Relationships
print("="*60)
print("Strong Relationships")
print("="*60)

for index,row in correlation_df.iterrows():

    if abs(row["Correlation"]) >= 0.7:

        print(f"{row['Feature']} --> Strong")

    elif abs(row["Correlation"]) >= 0.5:

        print(f"{row['Feature']} --> Moderate")

    else:

        print(f"{row['Feature']} --> Weak")



#Time-Series Analysis
os.makedirs("outputs/plots/time_series", exist_ok=True)
df["Year"] = df["datetime"].dt.year
df["Month"] = df["datetime"].dt.month
df["Day"] = df["datetime"].dt.day
df["Hour"] = df["datetime"].dt.hour

print("Time Features Created Successfully")

#Monthly PM2.5 Trend
monthly_pm25 = df.groupby("Month")["PM2.5"].mean().reset_index()

plt.figure(figsize=(10,5))

sns.lineplot(
    data=monthly_pm25,
    x="Month",
    y="PM2.5",
    marker="o"
)

plt.title("Average Monthly PM2.5")
plt.xlabel("Month")
plt.ylabel("PM2.5")

plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/plots/time_series/monthly_pm25.png")

plt.show()

#Yearly PM2.5 Trend
yearly_pm25 = df.groupby("Year")["PM2.5"].mean().reset_index()

plt.figure(figsize=(8,5))

sns.lineplot(
    data=yearly_pm25,
    x="Year",
    y="PM2.5",
    marker="o"
)

plt.title("Average Yearly PM2.5")
plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/plots/time_series/yearly_pm25.png")

plt.show()

#Hourly PM2.5 Trend
hourly_pm25 = df.groupby("Hour")["PM2.5"].mean().reset_index()

plt.figure(figsize=(10,5))

sns.lineplot(
    data=hourly_pm25,
    x="Hour",
    y="PM2.5",
    marker="o"
)

plt.title("Average Hourly PM2.5")
plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/plots/time_series/hourly_pm25.png")

plt.show()

#Monthly PM10 Trend
monthly_pm10 = df.groupby("Month")["PM10"].mean().reset_index()

plt.figure(figsize=(10,5))

sns.lineplot(
    data=monthly_pm10,
    x="Month",
    y="PM10",
    marker="o"
)

plt.title("Average Monthly PM10")
plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/plots/time_series/monthly_pm10.png")

plt.show()

#Monthly CO Trend
monthly_co = df.groupby("Month")["CO"].mean().reset_index()

plt.figure(figsize=(10,5))

sns.lineplot(
    data=monthly_co,
    x="Month",
    y="CO",
    marker="o"
)

plt.title("Average Monthly CO")
plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/plots/time_series/monthly_co.png")

plt.show()

#Monthly O₃ Trend
monthly_o3 = df.groupby("Month")["O3"].mean().reset_index()

plt.figure(figsize=(10,5))

sns.lineplot(
    data=monthly_o3,
    x="Month",
    y="O3",
    marker="o"
)

plt.title("Average Monthly O3")
plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/plots/time_series/monthly_o3.png")

plt.show()

#Save Trend Reports
monthly_pm25.to_csv(
    "outputs/reports/monthly_pm25.csv",
    index=False
)

yearly_pm25.to_csv(
    "outputs/reports/yearly_pm25.csv",
    index=False
)

hourly_pm25.to_csv(
    "outputs/reports/hourly_pm25.csv",
    index=False
)

print("Trend Reports Saved Successfully")

#summary
print("=" * 60)
print("Trend Summary")
print("=" * 60)

print("\nHighest Average Monthly PM2.5")
print(monthly_pm25.loc[monthly_pm25["PM2.5"].idxmax()])

print("\nLowest Average Monthly PM2.5")
print(monthly_pm25.loc[monthly_pm25["PM2.5"].idxmin()])

print("\nHighest Average Hourly PM2.5")
print(hourly_pm25.loc[hourly_pm25["PM2.5"].idxmax()])

print("\nLowest Average Hourly PM2.5")
print(hourly_pm25.loc[hourly_pm25["PM2.5"].idxmin()])





#Station-wise & Seasonal Analysis

os.makedirs("outputs/plots/station_season", exist_ok=True)


#Average PM2.5 by Station
station_pm25 = df.groupby("station")["PM2.5"].mean().sort_values()

plt.figure(figsize=(12,6))

station_pm25.plot(kind="bar")

plt.title("Average PM2.5 by Station")
plt.xlabel("Station")
plt.ylabel("Average PM2.5")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig("outputs/plots/station_season/station_pm25.png")

plt.show()

#Average PM10 by Station
station_pm10 = df.groupby("station")["PM10"].mean().sort_values()

plt.figure(figsize=(12,6))

station_pm10.plot(kind="bar")

plt.title("Average PM10 by Station")
plt.xlabel("Station")
plt.ylabel("Average PM10")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig("outputs/plots/station_season/station_pm10.png")

plt.show()

#Average CO by Station
station_co = df.groupby("station")["CO"].mean().sort_values()

plt.figure(figsize=(12,6))

station_co.plot(kind="bar")

plt.title("Average CO by Station")
plt.xlabel("Station")
plt.ylabel("Average CO")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig("outputs/plots/station_season/station_co.png")

plt.show()


#Save Station Report
station_report = df.groupby("station")[["PM2.5","PM10","CO"]].mean()

station_report.to_csv(
    "outputs/reports/station_report.csv"
)

print("Station Report Saved Successfully")


#Seasonal Analysis
def get_season(month):

    if month in [12,1,2]:
        return "Winter"

    elif month in [3,4,5]:
        return "Spring"

    elif month in [6,7,8]:
        return "Summer"

    else:
        return "Autumn"

df["Season"] = df["Month"].apply(get_season)

print(df["Season"].value_counts())

#PM2.5 by Season
season_pm25 = df.groupby("Season")["PM2.5"].mean()

plt.figure(figsize=(8,5))

season_pm25.plot(
    kind="bar"
)

plt.title("Average PM2.5 by Season")

plt.ylabel("PM2.5")

plt.tight_layout()

plt.savefig(
    "outputs/plots/station_season/season_pm25.png"
)

plt.show()

#PM10 by Season
season_pm10 = df.groupby("Season")["PM10"].mean()

plt.figure(figsize=(8,5))

season_pm10.plot(
    kind="bar"
)

plt.title("Average PM10 by Season")

plt.ylabel("PM10")

plt.tight_layout()

plt.savefig(
    "outputs/plots/station_season/season_pm10.png"
)

plt.show()

#CO by Season
season_co = df.groupby("Season")["CO"].mean()

plt.figure(figsize=(8,5))

season_co.plot(
    kind="bar"
)

plt.title("Average CO by Season")

plt.ylabel("CO")

plt.tight_layout()

plt.savefig(
    "outputs/plots/station_season/season_co.png"
)

plt.show()

#Save Seasonal Report
season_report = df.groupby("Season")[["PM2.5","PM10","CO"]].mean()

season_report.to_csv(
    "outputs/reports/season_report.csv"
)

print("Season Report Saved Successfully")

#Summary
print("="*60)
print("Station-wise Summary")
print("="*60)

print(station_report)

print("\n")

print("="*60)
print("Season-wise Summary")
print("="*60)

print(season_report)




#correlation Analysis & Feature Selection
os.makedirs("outputs/plots/correlation", exist_ok=True)
numerical_columns = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM"
]

#Correlation Matrix
correlation_matrix = df[numerical_columns].corr()
print(correlation_matrix)

#Save Correlation Matrix

correlation_matrix.to_csv(
    "outputs/reports/correlation_matrix.csv"
)
print("Correlation Matrix Saved Successfully")

#Correlation Heatmap
plt.figure(figsize=(12,10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "outputs/plots/correlation/correlation_heatmap.png"
)

plt.show()

#Correlation with PM2.5
pm25_corr = correlation_matrix["PM2.5"].sort_values(
    ascending=False
)

print(pm25_corr)
pm25_corr.to_csv(
    "outputs/reports/pm25_correlation.csv"
)

print("PM2.5 Correlation Saved")

#Feature Importance Table
feature_importance = pd.DataFrame({

    "Feature": pm25_corr.index,

    "Correlation": pm25_corr.values

})

print(feature_importance)

feature_importance.to_csv(

    "outputs/reports/feature_importance.csv",

    index=False

)

print("Feature Importance Saved")

#Strong Feature Selection
selected_features = feature_importance[
    abs(feature_importance["Correlation"]) >= 0.30
]

print(selected_features)

#Save Selected Features
selected_features.to_csv(
    "outputs/reports/selected_features.csv",
    index=False

)
print("Selected Features Saved")

#Feature Importance Plot
plt.figure(figsize=(10,6))

sns.barplot(

    data=selected_features,

    x="Correlation",

    y="Feature"

)

plt.title("Selected Features for PM2.5 Prediction")

plt.tight_layout()

plt.savefig(

    "outputs/plots/correlation/selected_features.png"

)
plt.show()
#summary
print("=" * 60)
print("Final Selected Features")
print("=" * 60)

for feature in selected_features["Feature"]:

    print(feature)
