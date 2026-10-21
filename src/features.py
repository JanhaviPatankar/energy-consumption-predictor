from pathlib import Path
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "household_power_hourly.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD HOURLY DATA
# ============================================================

print("Loading hourly dataset...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["timestamp"]
)

df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Original hourly dataset shape: {df.shape}")


# ============================================================
# 3. CREATE TARGET
# ============================================================

# We want to predict the NEXT hour's consumption.
#
# For a row representing hour t:
#   features = information available up to hour t
#   target   = consumption during hour t+1

df["next_hour_kwh"] = df["active_kwh"].shift(-1)


# ============================================================
# 4. CREATE TARGET-TIME CALENDAR FEATURES
# ============================================================

# These describe the hour we are trying to predict.
# They do NOT use future energy measurements.

target_timestamp = df["timestamp"] + pd.Timedelta(hours=1)

df["target_hour"] = target_timestamp.dt.hour
df["target_day_of_week"] = target_timestamp.dt.dayofweek
df["target_month"] = target_timestamp.dt.month
df["target_is_weekend"] = (
    target_timestamp.dt.dayofweek >= 5
).astype(int)


# ============================================================
# 5. CREATE ENERGY LAG FEATURES
# ============================================================

# Current hour t
df["active_kwh_lag_0"] = df["active_kwh"]

# Previous hour t-1
df["active_kwh_lag_1"] = df["active_kwh"].shift(1)

# Two hours earlier t-2
df["active_kwh_lag_2"] = df["active_kwh"].shift(2)

# Same hour approximately one day earlier
df["active_kwh_lag_23"] = df["active_kwh"].shift(23)

# Same hour one week earlier.
# The finalized project specification uses a lag of 167 hours.
df["active_kwh_lag_167"] = df["active_kwh"].shift(167)


# ============================================================
# 6. CREATE ROLLING FEATURES
# ============================================================

# Mean of current hour and previous 2 hours
df["active_kwh_mean_3h"] = (
    df["active_kwh"]
    .rolling(window=3, min_periods=3)
    .mean()
)

# Mean of current hour and previous 23 hours
df["active_kwh_mean_24h"] = (
    df["active_kwh"]
    .rolling(window=24, min_periods=24)
    .mean()
)


# ============================================================
# 7. SUB-METER LAG FEATURES
# ============================================================

df["sub1_kwh_lag_0"] = df["sub_metering_1_kwh"]
df["sub2_kwh_lag_0"] = df["sub_metering_2_kwh"]
df["sub3_kwh_lag_0"] = df["sub_metering_3_kwh"]


# ============================================================
# 8. SUB-METER 24-HOUR ROLLING FEATURES
# ============================================================

df["sub1_kwh_mean_24h"] = (
    df["sub_metering_1_kwh"]
    .rolling(window=24, min_periods=24)
    .mean()
)

df["sub2_kwh_mean_24h"] = (
    df["sub_metering_2_kwh"]
    .rolling(window=24, min_periods=24)
    .mean()
)

df["sub3_kwh_mean_24h"] = (
    df["sub_metering_3_kwh"]
    .rolling(window=24, min_periods=24)
    .mean()
)


# ============================================================
# 9. SELECT FINAL FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "target_hour",
    "target_day_of_week",
    "target_month",
    "target_is_weekend",

    "active_kwh_lag_0",
    "active_kwh_lag_1",
    "active_kwh_lag_2",
    "active_kwh_lag_23",
    "active_kwh_lag_167",

    "active_kwh_mean_3h",
    "active_kwh_mean_24h",

    "sub1_kwh_lag_0",
    "sub2_kwh_lag_0",
    "sub3_kwh_lag_0",

    "sub1_kwh_mean_24h",
    "sub2_kwh_mean_24h",
    "sub3_kwh_mean_24h",
]

TARGET_COLUMN = "next_hour_kwh"


# ============================================================
# 10. KEEP ONLY REQUIRED COLUMNS
# ============================================================

model_df = df[
    ["timestamp"] + FEATURE_COLUMNS + [TARGET_COLUMN]
].copy()


print("\nMissing values before removing incomplete rows:")
print(model_df.isnull().sum())


# ============================================================
# 11. REMOVE ROWS WITH MISSING FEATURES/TARGET
# ============================================================

# We do NOT interpolate these values.
# A forecasting row is kept only when every required
# feature and its target are available.

before_drop = len(model_df)

model_df = model_df.dropna(
    subset=FEATURE_COLUMNS + [TARGET_COLUMN]
).reset_index(drop=True)

after_drop = len(model_df)

print(
    f"\nRows removed because of missing "
    f"features/target: {before_drop - after_drop:,}"
)

print(f"Final forecasting dataset shape: {model_df.shape}")


# ============================================================
# 12. CHRONOLOGICAL 70/15/15 SPLIT
# ============================================================

total_rows = len(model_df)

train_end = int(total_rows * 0.70)
validation_end = int(total_rows * 0.85)

train_df = model_df.iloc[:train_end].copy()

validation_df = model_df.iloc[
    train_end:validation_end
].copy()

test_df = model_df.iloc[
    validation_end:
].copy()


# ============================================================
# 13. PRINT SPLIT INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("CHRONOLOGICAL DATA SPLIT")
print("=" * 60)

print(
    f"\nTraining rows:   {len(train_df):,}"
)
print(
    f"Validation rows: {len(validation_df):,}"
)
print(
    f"Test rows:       {len(test_df):,}"
)

print("\nTraining period:")
print(train_df["timestamp"].min())
print("to")
print(train_df["timestamp"].max())

print("\nValidation period:")
print(validation_df["timestamp"].min())
print("to")
print(validation_df["timestamp"].max())

print("\nTest period:")
print(test_df["timestamp"].min())
print("to")
print(test_df["timestamp"].max())


# ============================================================
# 14. SAVE DATASETS
# ============================================================

FEATURE_FILE = OUTPUT_DIR / "forecasting_features.csv"
TRAIN_FILE = OUTPUT_DIR / "train.csv"
VALIDATION_FILE = OUTPUT_DIR / "validation.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"

model_df.to_csv(
    FEATURE_FILE,
    index=False
)

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

validation_df.to_csv(
    VALIDATION_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)


# ============================================================
# 15. DISPLAY SAMPLE
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE FORECASTING ROWS")
print("=" * 60)

print(
    model_df[
        ["timestamp"]
        + FEATURE_COLUMNS[:8]
        + [TARGET_COLUMN]
    ].head(5).to_string(index=False)
)


print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print(f"\nSaved feature dataset:")
print(FEATURE_FILE)

print("\nSaved training dataset:")
print(TRAIN_FILE)

print("\nSaved validation dataset:")
print(VALIDATION_FILE)

print("\nSaved test dataset:")
print(TEST_FILE)