from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

MODEL_FILE = MODEL_DIR / "model.joblib"
TEST_FILE = DATA_DIR / "test.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MODEL AND TEST DATA
# ============================================================

print("=" * 60)
print("MODEL EXPLAINABILITY")
print("=" * 60)

print("\nLoading model...")

model = joblib.load(MODEL_FILE)

test_df = pd.read_csv(
    TEST_FILE,
    parse_dates=["timestamp"]
)


# ============================================================
# FEATURES AND TARGET
# ============================================================

FEATURES = [
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

TARGET = "next_hour_kwh"


X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

print("\nCalculating permutation feature importance...")

result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="neg_mean_absolute_error",
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)


# ============================================================
# CREATE IMPORTANCE TABLE
# ============================================================

importance_df = pd.DataFrame(
    {
        "Feature": FEATURES,
        "Importance": result.importances_mean,
        "Std": result.importances_std,
    }
)

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nFeature importance ranking:")
print("=" * 60)

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

importance_file = (
    MODEL_DIR / "feature_importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False
)

print(
    f"\nSaved feature importance table to:\n"
    f"{importance_file}"
)


# ============================================================
# CREATE FEATURE IMPORTANCE CHART
# ============================================================

top_features = importance_df.head(10).copy()

top_features = top_features.sort_values(
    by="Importance"
)


plt.figure(
    figsize=(10, 6)
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel(
    "Mean decrease in model performance when feature is shuffled"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 10 Feature Importance - HistGradientBoosting"
)

plt.tight_layout()


figure_file = (
    FIGURES_DIR
    / "feature_importance.png"
)

plt.savefig(
    figure_file,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(
    f"Saved feature importance chart to:\n"
    f"{figure_file}"
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("EXPLAINABILITY ANALYSIS COMPLETED")
print("=" * 60)