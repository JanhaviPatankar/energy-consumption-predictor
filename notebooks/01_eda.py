
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Locate the processed dataset
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "household_power_hourly.csv"
FIGURE_DIR = ROOT / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Load the dataset
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# Keep valid hourly consumption records
df = df.dropna(subset=["active_kwh"]).copy()

# Extract time features for analysis
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_name()
df["month"] = df["timestamp"].dt.month
df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5

sns.set_theme(style="whitegrid")

# 1. Average consumption by hour
hourly = df.groupby("hour")["active_kwh"].mean()

plt.figure(figsize=(10, 5))
hourly.plot(kind="bar")
plt.title("Average Household Consumption by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Energy (kWh)")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "hourly_consumption.png")
plt.show()

# 2. Weekday vs weekend
weekend = df.groupby("is_weekend")["active_kwh"].mean()
weekend.index = ["Weekday", "Weekend"]

plt.figure(figsize=(7, 5))
weekend.plot(kind="bar")
plt.title("Weekday vs Weekend Consumption")
plt.ylabel("Average Energy (kWh)")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "weekday_weekend.png")
plt.show()

# 3. Average monthly consumption
monthly = df.groupby("month")["active_kwh"].mean()

plt.figure(figsize=(10, 5))
monthly.plot(marker="o")
plt.title("Average Monthly Consumption")
plt.xlabel("Month")
plt.ylabel("Average Energy (kWh)")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.savefig(FIGURE_DIR / "monthly_consumption.png")
plt.show()

# 4. Distribution of hourly consumption
plt.figure(figsize=(10, 5))
sns.histplot(df["active_kwh"], bins=50)
plt.title("Distribution of Hourly Energy Consumption")
plt.xlabel("Energy (kWh)")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "consumption_distribution.png")
plt.show()

print("\nEDA completed successfully!")
print("Figures saved to:", FIGURE_DIR)
