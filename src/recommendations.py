import pandas as pd
import numpy as np


# ============================================================
# LOAD TRAINING DATA AND CALCULATE RECOMMENDATION THRESHOLDS
# ============================================================

def calculate_thresholds(train_path="data/processed/train.csv"):
    """
    Calculate thresholds using training data only.

    These thresholds are later used by the recommendation engine.
    """

    train_df = pd.read_csv(train_path)

    thresholds = {
        # Low / Medium / High consumption classification
        "low_medium": train_df["next_hour_kwh"].quantile(0.33),
        "medium_high": train_df["next_hour_kwh"].quantile(0.67),

        # High sub-meter usage thresholds
        "sub1_high": train_df["sub1_kwh_lag_0"].quantile(0.90),
        "sub2_high": train_df["sub2_kwh_lag_0"].quantile(0.90),
        "sub3_high": train_df["sub3_kwh_lag_0"].quantile(0.90),
    }

    return thresholds


# ============================================================
# CLASSIFY PREDICTED CONSUMPTION
# ============================================================

def classify_consumption(predicted_kwh, thresholds):
    """
    Classify predicted consumption as Low, Medium, or High.
    """

    if predicted_kwh < thresholds["low_medium"]:
        return "Low"

    elif predicted_kwh < thresholds["medium_high"]:
        return "Medium"

    else:
        return "High"


# ============================================================
# GENERATE ENERGY-SAVING RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    predicted_kwh,
    current_sub1,
    current_sub2,
    current_sub3,
    thresholds,
    peak_start=None,
    peak_end=None,
    forecast_hour=None
):
    """
    Generate transparent rule-based energy-saving recommendations.

    Parameters
    ----------
    predicted_kwh : float
        Predicted consumption for the next hour.

    current_sub1 : float
        Current-hour Sub-metering 1 usage.

    current_sub2 : float
        Current-hour Sub-metering 2 usage.

    current_sub3 : float
        Current-hour Sub-metering 3 usage.

    thresholds : dict
        Thresholds calculated from training data.

    peak_start : int, optional
        User-configured peak window starting hour.

    peak_end : int, optional
        User-configured peak window ending hour.

    forecast_hour : int, optional
        Hour for which the next-hour prediction is being made.
    """

    recommendations = []

    # --------------------------------------------------------
    # Consumption classification
    # --------------------------------------------------------

    level = classify_consumption(predicted_kwh, thresholds)

    # --------------------------------------------------------
    # Rule 1: High predicted consumption
    # --------------------------------------------------------

    if level == "High":
        recommendations.append(
            "Predicted consumption is high. Consider reducing or "
            "deferring non-essential high-energy activities to a "
            "lower-consumption period."
        )

    # --------------------------------------------------------
    # Rule 2: Sub-meter 3
    # --------------------------------------------------------

    if current_sub3 > thresholds["sub3_high"]:
        recommendations.append(
            "Sub-meter 3 usage is unusually high. Consider reviewing "
            "air-conditioning or water-heater usage. Note: Sub-meter 3 "
            "combines these loads, so this reading does not identify "
            "a specific appliance."
        )

    # --------------------------------------------------------
    # Rule 3: Sub-meter 1
    # --------------------------------------------------------

    if current_sub1 > thresholds["sub1_high"]:
        recommendations.append(
            "Sub-meter 1 usage is unusually high. Consider reviewing "
            "high-load kitchen activities and switching off equipment "
            "that is not currently needed."
        )

    # --------------------------------------------------------
    # Rule 4: Sub-meter 2
    # --------------------------------------------------------

    if current_sub2 > thresholds["sub2_high"]:
        recommendations.append(
            "Sub-meter 2 usage is unusually high. Consider reviewing "
            "laundry-related high-energy activities and avoiding "
            "unnecessary operation."
        )

    # --------------------------------------------------------
    # Rule 5: Optional user-configured peak period
    # --------------------------------------------------------

    if (
        peak_start is not None
        and peak_end is not None
        and forecast_hour is not None
    ):
        if peak_start <= peak_end:
            is_peak = peak_start <= forecast_hour < peak_end
        else:
            # Handles windows crossing midnight.
            is_peak = (
                forecast_hour >= peak_start
                or forecast_hour < peak_end
            )

        if is_peak:
            recommendations.append(
                "This prediction falls within your configured peak "
                "period. If possible, consider postponing flexible "
                "high-energy activities."
            )

    # --------------------------------------------------------
    # Default recommendation
    # --------------------------------------------------------

    if not recommendations:
        recommendations.append(
            "Predicted consumption is not unusually high based on "
            "the current thresholds. Continue monitoring usage and "
            "switch off appliances that are not needed."
        )

    return level, recommendations


# ============================================================
# TEST THE RECOMMENDATION ENGINE
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ENERGY-SAVING RECOMMENDATION ENGINE")
    print("=" * 60)

    # Calculate thresholds from training data
    thresholds = calculate_thresholds()

    print("\nTraining-derived thresholds:")
    print(f"Low/Medium threshold : {thresholds['low_medium']:.4f} kWh")
    print(f"Medium/High threshold: {thresholds['medium_high']:.4f} kWh")
    print(f"Sub-meter 1 high     : {thresholds['sub1_high']:.4f} kWh")
    print(f"Sub-meter 2 high     : {thresholds['sub2_high']:.4f} kWh")
    print(f"Sub-meter 3 high     : {thresholds['sub3_high']:.4f} kWh")

    # Example prediction
    example_prediction = thresholds["medium_high"] + 0.5

    # Example current sub-meter values
    example_sub1 = thresholds["sub1_high"] + 0.1
    example_sub2 = thresholds["sub2_high"] - 0.1
    example_sub3 = thresholds["sub3_high"] + 0.1

    level, recommendations = generate_recommendations(
        predicted_kwh=example_prediction,
        current_sub1=example_sub1,
        current_sub2=example_sub2,
        current_sub3=example_sub3,
        thresholds=thresholds,
        forecast_hour=19,
        peak_start=18,
        peak_end=22
    )

    print("\nExample prediction:")
    print(f"Predicted next-hour consumption: {example_prediction:.4f} kWh")
    print(f"Consumption level: {level}")

    print("\nRecommendations:")

    for i, recommendation in enumerate(recommendations, start=1):
        print(f"{i}. {recommendation}")

    print("\n" + "=" * 60)
    print("RECOMMENDATION ENGINE TEST COMPLETED")
    print("=" * 60)