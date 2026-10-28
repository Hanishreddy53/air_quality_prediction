import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
path = "data/raw"

files = os.listdir(path)

print(files)
all_data = []

for file in files:

    if file.endswith(".csv"):

        df = pd.read_csv(os.path.join(path,file))

        station_name = file.split(".")[0]

        df["station"] = station_name

        all_data.append(df)

combined_df = pd.concat(all_data,ignore_index=True)

# Create processed folder if it doesn't exist
os.makedirs("data/processed", exist_ok=True)

combined_df.to_csv("data/processed/combined_dataset.csv", index=False)

print("Combined dataset saved successfully!")

# Load the combined dataset for analysis
df = pd.read_csv("data/processed/combined_dataset.csv")

#shape,size and head of the combined dataset
print(combined_df.head(5))
print(combined_df.shape,combined_df.size)

#Display Column Names
print(combined_df.columns)

#Dataset Information
df.info()

#Data Types
print(df.dtypes)

#Statistical Summary
print(df.describe())

#checking missing values 
missing = df.isnull().sum()
print(missing)
missing_percent = (df.isnull().sum()/len(df))*100
print(missing_percent.sort_values(ascending=False))

#checking for duplicates
duplicates = df.duplicated().sum()
print("Duplicate Rows :", duplicates)

#checking unique values in each column
print(df.nunique())

#Check Wind Directions
print(df["wd"].unique())

#Check Stations
print(df["station"].unique())

#Records per Station
print(df["station"].value_counts())

#Convert Date and Time
df["datetime"] = pd.to_datetime(df[["year","month","day","hour"]])
print(df["datetime"].head())

#Date Range
print("Start Date :", df["datetime"].min())
print("End Date :", df["datetime"].max())

#Pollutant Columns
pollutants = ["PM2.5","PM10","SO2","NO2","CO","O3"]
print(df[pollutants].describe().T)

#Weather Columns
weather = ["TEMP","PRES","DEWP","RAIN","WSPM"]
print(df[weather].describe().T)

#saving dataset report
report = pd.DataFrame({
    "Data Type": df.dtypes,
    "Missing Values": df.isnull().sum(),
    "Unique Values": df.nunique()
})


