import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "smart_logistics_dataset.csv"


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("SMART LOGISTICS DATASET INSPECTION")
print("=" * 60)

print(f"\nDataset path:")
print(DATA_PATH)

if not DATA_PATH.exists():
    print("\n❌ ERROR: Dataset file not found!")
    print("\nPlease make sure this file exists:")
    print(DATA_PATH)
    raise SystemExit


df = pd.read_csv(DATA_PATH)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET SHAPE")
print("=" * 60)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# COLUMN NAMES
# ============================================================

print("\n" + "=" * 60)
print("COLUMNS")
print("=" * 60)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")


# ============================================================
# FIRST 5 ROWS
# ============================================================

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)

print(df.head().to_string())


# ============================================================
# DATA TYPES
# ============================================================

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()

print(missing)

print(f"\nTotal missing values: {missing.sum()}")


# ============================================================
# DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

print(f"Duplicate rows: {df.duplicated().sum()}")


# ============================================================
# UNIQUE VALUES
# ============================================================

print("\n" + "=" * 60)
print("UNIQUE VALUES")
print("=" * 60)

for column in df.columns:

    print(f"\n--- {column} ---")

    unique_values = df[column].unique()

    print(unique_values[:20])

    if len(unique_values) > 20:
        print(f"... and {len(unique_values) - 20} more")


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

if "Logistics_Delay" in df.columns:

    print("\nValue counts:")

    print(df["Logistics_Delay"].value_counts())

    print("\nPercentage:")

    print(
        df["Logistics_Delay"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

else:

    print("⚠ Logistics_Delay column was not found.")


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)

print(df.describe().to_string())


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("DATASET INSPECTION COMPLETED")
print("=" * 60)