from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD RESULTS
# ============================================================

results_file = MODEL_DIR / "model_results.csv"
predictions_file = MODEL_DIR / "test_predictions.csv"

results = pd.read_csv(results_file)
predictions = pd.read_csv(
    predictions_file,
    parse_dates=["timestamp"]
)

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("\nModel comparison:")
print(results.to_string(index=False))


# ============================================================
# 3. IDENTIFY BEST MODEL
# ============================================================

best_model = results.sort_values(
    by="MAE"
).iloc[0]

print("\nBest model:")
print(best_model["Model"])

print(f"Validation MAE: {best_model['MAE']:.4f} kWh")
print(f"Validation RMSE: {best_model['RMSE']:.4f} kWh")
print(f"Validation R²: {best_model['R2']:.4f}")


# ============================================================
# 4. ACTUAL VS PREDICTED VALUES
# ============================================================

predictions["error"] = (
    predictions["actual_kwh"]
    - predictions["predicted_kwh"]
)

predictions["absolute_error"] = (
    predictions["error"].abs()
)


# ============================================================
# 5. ACTUAL VS PREDICTED LINE CHART
# ============================================================

# Plot only the first 168 test hours (one week)
# so that the graph remains readable.

plot_data = predictions.head(168)

plt.figure(figsize=(14, 6))

plt.plot(
    plot_data["timestamp"],
    plot_data["actual_kwh"],
    label="Actual",
    linewidth=2
)

plt.plot(
    plot_data["timestamp"],
    plot_data["predicted_kwh"],
    label="Predicted",
    linewidth=2
)

plt.title(
    "Actual vs Predicted Household Energy Consumption"
)

plt.xlabel("Time")
plt.ylabel("Energy Consumption (kWh)")

plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "actual_vs_predicted.png",
    dpi=300
)

plt.show()


# ============================================================
# 6. ACTUAL VS PREDICTED SCATTER PLOT
# ============================================================

plt.figure(figsize=(8, 8))

sns.scatterplot(
    data=predictions.sample(
        min(3000, len(predictions)),
        random_state=42
    ),
    x="actual_kwh",
    y="predicted_kwh",
    alpha=0.5
)

# Perfect prediction reference line
min_value = min(
    predictions["actual_kwh"].min(),
    predictions["predicted_kwh"].min()
)

max_value = max(
    predictions["actual_kwh"].max(),
    predictions["predicted_kwh"].max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Perfect Prediction"
)

plt.title(
    "Actual vs Predicted Energy Consumption"
)

plt.xlabel("Actual Energy (kWh)")
plt.ylabel("Predicted Energy (kWh)")

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "actual_vs_predicted_scatter.png",
    dpi=300
)

plt.show()


# ============================================================
# 7. PREDICTION ERROR OVER TIME
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(
    plot_data["timestamp"],
    plot_data["error"],
    linewidth=1.5
)

plt.axhline(
    0,
    linestyle="--"
)

plt.title(
    "Prediction Error Over Time"
)

plt.xlabel("Time")
plt.ylabel("Prediction Error (kWh)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "prediction_error.png",
    dpi=300
)

plt.show()


# ============================================================
# 8. MODEL COMPARISON — MAE
# ============================================================

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results,
    x="Model",
    y="MAE"
)

plt.title(
    "Model Comparison - Mean Absolute Error"
)

plt.xlabel("Model")
plt.ylabel("MAE (kWh)")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "model_comparison_mae.png",
    dpi=300
)

plt.show()


# ============================================================
# 9. MODEL COMPARISON — R²
# ============================================================

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results,
    x="Model",
    y="R2"
)

plt.title(
    "Model Comparison - R² Score"
)

plt.xlabel("Model")
plt.ylabel("R² Score")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "model_comparison_r2.png",
    dpi=300
)

plt.show()


# ============================================================
# 10. ERROR SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TEST PREDICTION ERROR SUMMARY")
print("=" * 60)

print(
    f"Mean absolute error: "
    f"{predictions['absolute_error'].mean():.4f} kWh"
)

print(
    f"Maximum absolute error: "
    f"{predictions['absolute_error'].max():.4f} kWh"
)

print(
    f"Mean prediction error: "
    f"{predictions['error'].mean():.4f} kWh"
)


# ============================================================
# 11. SAVE ERROR-ENHANCED PREDICTIONS
# ============================================================

predictions.to_csv(
    MODEL_DIR / "test_predictions_with_errors.csv",
    index=False
)


print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print(
    f"\nGraphs saved to:\n{FIGURE_DIR}"
)