from pathlib import Path
import json

import joblib
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE = MODEL_DIR / "model.joblib"
METADATA_FILE = MODEL_DIR / "metadata.json"
FEATURE_FILE = DATA_DIR / "forecasting_features.csv"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the trained forecasting model.
    """

    return joblib.load(MODEL_FILE)


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():
    """
    Load model metadata and thresholds.
    """

    with open(METADATA_FILE, "r") as f:
        return json.load(f)


# ============================================================
# LOAD FORECASTING DATA
# ============================================================

def load_forecasting_data():
    """
    Load the engineered forecasting dataset.
    """

    df = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["timestamp"]
    )

    return df


# ============================================================
# GET VALID FORECAST ORIGINS
# ============================================================

def get_valid_forecast_origins():
    """
    Return timestamps for which all required prediction
    features are available.
    """

    df = load_forecasting_data()
    metadata = load_metadata()

    features = metadata["features"]

    valid_df = df.dropna(subset=features).copy()

    return valid_df["timestamp"].sort_values().tolist()


# ============================================================
# PREPARE INPUT FEATURES
# ============================================================

def prepare_features(timestamp):
    """
    Retrieve the feature row corresponding to a forecast origin.
    """

    df = load_forecasting_data()
    metadata = load_metadata()

    features = metadata["features"]

    timestamp = pd.Timestamp(timestamp)

    matching_rows = df[df["timestamp"] == timestamp]

    if matching_rows.empty:
        raise ValueError(
            f"No forecasting data found for timestamp: {timestamp}"
        )

    row = matching_rows.iloc[0]

    # Check that all required features are available.
    missing_features = [
        feature
        for feature in features
        if pd.isna(row[feature])
    ]

    if missing_features:
        raise ValueError(
            "Required features are missing for this timestamp: "
            + ", ".join(missing_features)
        )

    X = pd.DataFrame(
        [row[features].to_dict()],
        columns=features
    )

    return X


# ============================================================
# PREDICT NEXT-HOUR CONSUMPTION
# ============================================================

def predict_next_hour(timestamp):
    """
    Predict next-hour household electricity consumption.

    Returns
    -------
    predicted_kwh : float
        Predicted consumption for the next hour.

    level : str
        Low, Medium, or High classification.

    """

    model = load_model()
    metadata = load_metadata()

    X = prepare_features(timestamp)

    prediction = float(model.predict(X)[0])

    thresholds = metadata["classification_thresholds"]

    if prediction < thresholds["low_medium"]:
        level = "Low"

    elif prediction < thresholds["medium_high"]:
        level = "Medium"

    else:
        level = "High"

    return prediction, level


# ============================================================
# GET CURRENT SUB-METER VALUES
# ============================================================

def get_current_submeter_values(timestamp):
    """
    Return current-hour sub-meter values for recommendation rules.
    """

    df = load_forecasting_data()

    timestamp = pd.Timestamp(timestamp)

    matching_rows = df[df["timestamp"] == timestamp]

    if matching_rows.empty:
        raise ValueError(
            f"No data found for timestamp: {timestamp}"
        )

    row = matching_rows.iloc[0]

    return {
        "sub1": float(row["sub1_kwh_lag_0"]),
        "sub2": float(row["sub2_kwh_lag_0"]),
        "sub3": float(row["sub3_kwh_lag_0"])
    }


# ============================================================
# TEST THE PREDICTION PIPELINE
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PREDICTION PIPELINE TEST")
    print("=" * 60)

    # Get valid timestamps
    timestamps = get_valid_forecast_origins()

    print(f"\nValid forecast origins: {len(timestamps)}")

    # Use the first valid timestamp for testing
    test_timestamp = timestamps[0]

    print(f"Test timestamp: {test_timestamp}")

    # Make prediction
    predicted_kwh, level = predict_next_hour(
        test_timestamp
    )

    print(
        f"Predicted next-hour consumption: "
        f"{predicted_kwh:.4f} kWh"
    )

    print(f"Consumption level: {level}")

    # Get sub-meter values
    submeters = get_current_submeter_values(
        test_timestamp
    )

    print("\nCurrent sub-meter values:")

    print(
        f"Sub-meter 1: "
        f"{submeters['sub1']:.4f} kWh"
    )

    print(
        f"Sub-meter 2: "
        f"{submeters['sub2']:.4f} kWh"
    )

    print(
        f"Sub-meter 3: "
        f"{submeters['sub3']:.4f} kWh"
    )

    print("\n" + "=" * 60)
    print("PREDICTION PIPELINE TEST COMPLETED")
    print("=" * 60)