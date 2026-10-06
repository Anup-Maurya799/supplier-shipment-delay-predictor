import pandas as pd
import numpy as np

TARGET_COLUMN = "Logistics_Delay"


def load_data(file_path):
    return pd.read_csv(file_path)


def clean_data(df):
    df = df.copy().drop_duplicates()

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["Year"] = df["Timestamp"].dt.year
        df["Month"] = df["Timestamp"].dt.month
        df["Day"] = df["Timestamp"].dt.day
        df["Hour"] = df["Timestamp"].dt.hour
        df["Day_of_Week"] = df["Timestamp"].dt.dayofweek
        df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)
        df = df.drop(columns=["Timestamp"])

    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].astype(str).str.strip()
        df.loc[df[column].isin(["nan", "None", "NaT"]), column] = np.nan

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    if df[TARGET_COLUMN].dtype == "object":
        mapping = {"Late": 1, "Delayed": 1, "On-Time": 0, "On Time": 0, "OnTime": 0, "0": 0, "1": 1}
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map(mapping)

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def create_features(df):
    df = df.copy()

    if "Temperature" in df.columns:
        df["Temperature_Risk"] = pd.cut(
            df["Temperature"],
            bins=[-np.inf, 10, 25, 35, np.inf],
            labels=["Low", "Normal", "High", "Extreme"]
        )

    if "Humidity" in df.columns:
        df["Humidity_Risk"] = pd.cut(
            df["Humidity"],
            bins=[-np.inf, 30, 60, 80, np.inf],
            labels=["Low", "Normal", "High", "Very High"]
        )

    if "Waiting_Time" in df.columns:
        df["Waiting_Risk"] = pd.cut(
            df["Waiting_Time"],
            bins=[-np.inf, 20, 40, 60, np.inf],
            labels=["Low", "Medium", "High", "Critical"]
        )

    if "Inventory_Level" in df.columns:
        df["Inventory_Risk"] = pd.cut(
            df["Inventory_Level"],
            bins=[-np.inf, 250, 500, 750, np.inf],
            labels=["Low", "Medium", "High", "Very High"]
        )

    return df


def get_model_data(df):
    X = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    y = df[TARGET_COLUMN]

    # These fields describe the outcome/observed operating state and are not
    # available reliably before a shipment is completed. Keep them for
    # dashboard analytics, but exclude them from predictive ML features.
    leakage_columns = [
        "Shipment_Status",
        "Traffic_Status",
        "Logistics_Delay_Reason",
    ]

    X = X.drop(columns=[c for c in leakage_columns if c in X.columns], errors="ignore")
    return X, y
