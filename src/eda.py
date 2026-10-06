from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "smart_logistics_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA_PATH)


# ============================================================
# BASIC CLEANING
# ============================================================

df = df.drop_duplicates()

if "Timestamp" in df.columns:

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    df["Month"] = (
        df["Timestamp"]
        .dt.month
    )

    df["Month_Name"] = (
        df["Timestamp"]
        .dt.strftime("%b")
    )


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

processed_path = (
    OUTPUT_DIR
    / "shipment_data.csv"
)

df.to_csv(
    processed_path,
    index=False
)

print(
    f"Processed data saved to:\n"
    f"{processed_path}"
)


# ============================================================
# DELAY RATE
# ============================================================

if "Logistics_Delay" in df.columns:

    print("\nDelay distribution:")

    print(
        df["Logistics_Delay"]
        .value_counts()
    )


# ============================================================
# TRAFFIC ANALYSIS
# ============================================================

if (
    "Traffic_Status" in df.columns
    and "Logistics_Delay" in df.columns
):

    traffic_delay = (
        df.groupby("Traffic_Status")
        ["Logistics_Delay"]
        .mean()
        .sort_values(
            ascending=False
        )
        * 100
    )

    print("\nDelay rate by traffic:")

    print(
        traffic_delay
    )


# ============================================================
# SHIPMENT STATUS
# ============================================================

if "Shipment_Status" in df.columns:

    print(
        "\nShipment Status:"
    )

    print(
        df["Shipment_Status"]
        .value_counts()
    )


# ============================================================
# WEATHER SUMMARY
# ============================================================

if "Temperature" in df.columns:

    print(
        "\nTemperature statistics:"
    )

    print(
        df["Temperature"]
        .describe()
    )


if "Humidity" in df.columns:

    print(
        "\nHumidity statistics:"
    )

    print(
        df["Humidity"]
        .describe()
    )


print(
    "\nEDA preparation completed."
)