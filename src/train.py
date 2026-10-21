from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import json

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading datasets...")

train_df = pd.read_csv(
    TRAIN_FILE,
    parse_dates=["timestamp"]
)

validation_df = pd.read_csv(
    VALIDATION_FILE,
    parse_dates=["timestamp"]
)

test_df = pd.read_csv(
    TEST_FILE,
    parse_dates=["timestamp"]
)

print(f"Training shape:   {train_df.shape}")
print(f"Validation shape: {validation_df.shape}")
print(f"Test shape:       {test_df.shape}")


# ============================================================
# 3. DEFINE FEATURES AND TARGET
# ============================================================

TARGET = "next_hour_kwh"

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


X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_validation = validation_df[FEATURES]
y_validation = validation_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ============================================================
# 4. DEFINE FEATURE TYPES
# ============================================================

# Calendar features are categorical for Linear Regression.
CATEGORICAL_FEATURES = [
    "target_hour",
    "target_day_of_week",
    "target_month",
]

NUMERICAL_FEATURES = [
    feature
    for feature in FEATURES
    if feature not in CATEGORICAL_FEATURES
]


# ============================================================
# 5. PREPROCESSING FOR LINEAR REGRESSION
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        ),
    ]
)


numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
    ]
)


linear_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        ),
        (
            "numerical",
            numerical_pipeline,
            NUMERICAL_FEATURES
        ),
    ]
)


# ============================================================
# 6. LINEAR REGRESSION
# ============================================================

linear_model = Pipeline(
    steps=[
        (
            "preprocessor",
            linear_preprocessor
        ),
        (
            "model",
            LinearRegression()
        ),
    ]
)


# ============================================================
# 7. RANDOM FOREST
# ============================================================

random_forest = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=150,
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        ),
    ]
)


# ============================================================
# 8. HISTOGRAM GRADIENT BOOSTING
# ============================================================

hist_gradient_boosting = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.08,
                max_leaf_nodes=31,
                random_state=42
            )
        ),
    ]
)


models = {
    "Linear Regression": linear_model,
    "Random Forest": random_forest,
    "HistGradientBoosting": hist_gradient_boosting,
}


# ============================================================
# 9. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X, y):
    """
    Calculate regression evaluation metrics.
    """

    predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        predictions
    )

    mse = mean_squared_error(
        y,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y,
        predictions
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse,
        "R2": r2,
    }


# ============================================================
# 10. TRAIN AND VALIDATE MODELS
# ============================================================

validation_results = []

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train,
        y_train
    )

    metrics = evaluate_model(
        model,
        X_validation,
        y_validation
    )

    validation_results.append(
        {
            "Model": name,
            **metrics
        }
    )

    print(
        f"Validation MAE:  {metrics['MAE']:.4f} kWh"
    )

    print(
        f"Validation RMSE: {metrics['RMSE']:.4f} kWh"
    )

    print(
        f"Validation MSE:  {metrics['MSE']:.4f}"
    )

    print(
        f"Validation R²:   {metrics['R2']:.4f}"
    )


# ============================================================
# 11. VALIDATION COMPARISON
# ============================================================

validation_results_df = pd.DataFrame(
    validation_results
)

validation_results_df = validation_results_df.sort_values(
    by="MAE"
).reset_index(drop=True)


print("\n" + "=" * 60)
print("VALIDATION MODEL COMPARISON")
print("=" * 60)

print(
    validation_results_df.to_string(
        index=False
    )
)


# ============================================================
# 12. SELECT BEST MODEL
# ============================================================

best_model_name = validation_results_df.iloc[0]["Model"]

print(
    f"\nBest model according to validation MAE: "
    f"{best_model_name}"
)


# ============================================================
# 13. RETRAIN BEST MODEL ON TRAIN + VALIDATION
# ============================================================

print("\nRetraining selected model...")

X_train_validation = pd.concat(
    [
        X_train,
        X_validation
    ],
    axis=0
)

y_train_validation = pd.concat(
    [
        y_train,
        y_validation
    ],
    axis=0
)

best_model = models[best_model_name]

best_model.fit(
    X_train_validation,
    y_train_validation
)


# ============================================================
# 14. FINAL TEST EVALUATION
# ============================================================

test_metrics = evaluate_model(
    best_model,
    X_test,
    y_test
)


print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"Model: {best_model_name}"
)

print(
    f"Test MAE:  {test_metrics['MAE']:.4f} kWh"
)

print(
    f"Test RMSE: {test_metrics['RMSE']:.4f} kWh"
)

print(
    f"Test MSE:  {test_metrics['MSE']:.4f}"
)

print(
    f"Test R²:   {test_metrics['R2']:.4f}"
)


# ============================================================
# 15. SAVE MODEL
# ============================================================

MODEL_FILE = MODEL_DIR / "model.joblib"

joblib.dump(
    best_model,
    MODEL_FILE
)

print(
    f"\nSaved model to:\n{MODEL_FILE}"
)


# ============================================================
# 16. SAVE RESULTS
# ============================================================

results_file = MODEL_DIR / "model_results.csv"

validation_results_df.to_csv(
    results_file,
    index=False
)

print(
    f"Saved validation results to:\n{results_file}"
)


# ============================================================
# 17. SAVE TEST PREDICTIONS
# ============================================================

test_predictions = best_model.predict(
    X_test
)

prediction_df = pd.DataFrame(
    {
        "timestamp": test_df["timestamp"],
        "actual_kwh": y_test,
        "predicted_kwh": test_predictions,
    }
)

prediction_file = MODEL_DIR / "test_predictions.csv"

prediction_df.to_csv(
    prediction_file,
    index=False
)

print(
    f"Saved test predictions to:\n{prediction_file}"
)

# ============================================================
# 18. SAVE MODEL METADATA
# ============================================================

print("\nSaving model metadata...")

# Classification thresholds are calculated from the
# train + validation target data used to retrain the final model.
low_medium_threshold = y_train_validation.quantile(0.33)
medium_high_threshold = y_train_validation.quantile(0.67)

# Recommendation thresholds are calculated from training data only.
sub1_high_threshold = train_df["sub1_kwh_lag_0"].quantile(0.90)
sub2_high_threshold = train_df["sub2_kwh_lag_0"].quantile(0.90)
sub3_high_threshold = train_df["sub3_kwh_lag_0"].quantile(0.90)

metadata = {
    "selected_model": best_model_name,

    "target": TARGET,

    "features": FEATURES,

    "classification_thresholds": {
        "low_medium": float(low_medium_threshold),
        "medium_high": float(medium_high_threshold)
    },

    "recommendation_thresholds": {
        "sub1_high": float(sub1_high_threshold),
        "sub2_high": float(sub2_high_threshold),
        "sub3_high": float(sub3_high_threshold)
    },

    "validation_metrics": {
        "MAE": float(
            validation_results_df.iloc[0]["MAE"]
        ),
        "RMSE": float(
            validation_results_df.iloc[0]["RMSE"]
        ),
        "MSE": float(
            validation_results_df.iloc[0]["MSE"]
        ),
        "R2": float(
            validation_results_df.iloc[0]["R2"]
        )
    },

    "test_metrics": {
        "MAE": float(test_metrics["MAE"]),
        "RMSE": float(test_metrics["RMSE"]),
        "MSE": float(test_metrics["MSE"]),
        "R2": float(test_metrics["R2"])
    }
}

metadata_file = MODEL_DIR / "metadata.json"

with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Saved metadata to:\n{metadata_file}")

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)