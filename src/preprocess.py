from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_FILE = RAW_DIR / "household_power_consumption.txt"
OUTPUT_FILE = PROCESSED_DIR / "household_power_hourly.csv"

# Official UCI download
ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
ZIP_FILE = RAW_DIR / "household_power_consumption.zip"

# Readings needed for an hour to be considered sufficiently complete
MIN_VALID_MINUTES = 45

# Columns needed for this project’s hourly target and sub-meter context
MEASUREMENT_COLUMNS = [
    "Global_active_power",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def download_dataset_if_needed():
    """Download and extract the UCI file if it is not already present."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_FILE.exists():
        print(f"Raw dataset already exists: {RAW_FILE}")
        return

    print("Downloading the UCI dataset...")
    response = requests.get(ZIP_URL, timeout=120)
    response.raise_for_status()
    ZIP_FILE.write_bytes(response.content)

    print("Extracting the dataset...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        # The ZIP contains the dataset text file.
        matching_files = [
            name
            for name in zip_ref.namelist()
            if name.endswith("household_power_consumption.txt")
        ]
        if not matching_files:
            raise FileNotFoundError(
                "Could not find household_power_consumption.txt in the ZIP."
            )

        # Extract only the required file into data/raw.
        extracted_path = zip_ref.extract(matching_files[0], RAW_DIR)
        extracted_path = Path(extracted_path)

        # Put it at the expected path if it was extracted into a subfolder.
        if extracted_path != RAW_FILE:
            extracted_path.replace(RAW_FILE)

    print(f"Dataset extracted to: {RAW_FILE}")


def main():
    download_dataset_if_needed()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # The UCI file is semicolon-separated. The '?' marker means a missing value.
    print("\nLoading raw dataset...")
    raw = pd.read_csv(
        RAW_FILE,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )

    print(f"Raw shape: {raw.shape}")
    print("Raw columns:")
    print(raw.columns.tolist())

    # Combine Date and Time into one timestamp.
    # The source format uses day/month/year.
    raw["timestamp"] = pd.to_datetime(
        raw["Date"].astype(str) + " " + raw["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    # Convert the required measurements to numeric.
    # Invalid strings become NaN rather than causing the program to fail.
    for column in MEASUREMENT_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    # Report malformed timestamps, if any.
    invalid_timestamps = raw["timestamp"].isna().sum()
    print(f"\nRows with invalid timestamps: {invalid_timestamps}")

    # Remove rows where a timestamp could not be constructed, then index by time.
    raw = raw.dropna(subset=["timestamp"]).copy()
    raw = raw.set_index("timestamp").sort_index()

    # Basic data-quality and date-range checks.
    print(f"Rows after timestamp parsing: {len(raw):,}")
    print(f"Dataset date range: {raw.index.min()} to {raw.index.max()}")

    print("\nMissing readings in required measurement columns:")
    print(raw[MEASUREMENT_COLUMNS].isna().sum())

    # Count valid minute readings for each measurement in every clock hour.
    # A reading is valid when it is not NaN.
    valid_counts = raw[MEASUREMENT_COLUMNS].resample("h").count()

    # Compute hourly energy estimates:
    # - Global_active_power is in kW. For a complete-enough hour:
    #   mean kW over the hour × 1 hour = hourly kWh.
    # - Sub-meter channels are Wh per minute. For a complete-enough hour:
    #   mean Wh/min × 60 minutes ÷ 1000 = hourly kWh.
    hourly_mean = raw[MEASUREMENT_COLUMNS].resample("h").mean()

    hourly = pd.DataFrame(index=hourly_mean.index)

    for column in MEASUREMENT_COLUMNS:
        # Keep an hour only if that column has at least 45 valid minute readings.
        enough_coverage = valid_counts[column] >= MIN_VALID_MINUTES

        if column == "Global_active_power":
            hourly["active_kwh"] = hourly_mean[column].where(enough_coverage)
        else:
            output_column = column.lower() + "_kwh"
            hourly[output_column] = (
                hourly_mean[column] * 60 / 1000
            ).where(enough_coverage)

    # Save coverage counts too, so you can inspect the quality of each hourly value.
    # These are quality-control columns, not model features.
    for column in MEASUREMENT_COLUMNS:
        count_column = column.lower() + "_valid_minutes"
        hourly[count_column] = valid_counts[column].reindex(hourly.index).fillna(0).astype(int)

    # Report the coverage check.
    active_valid_hours = (valid_counts["Global_active_power"] >= MIN_VALID_MINUTES).sum()
    total_hour_bins = len(hourly)
    print(
        "\nGlobal active power hours meeting coverage threshold "
        f"(>= {MIN_VALID_MINUTES} valid minutes): "
        f"{active_valid_hours:,} / {total_hour_bins:,}"
    )

    # Save hourly dataset as CSV, retaining timestamp as a normal column.
    hourly.index.name = "timestamp"
    hourly.to_csv(OUTPUT_FILE)

    print(f"\nHourly dataset saved to: {OUTPUT_FILE}")
    print(f"Hourly dataset shape: {hourly.shape}")
    print("\nFirst five hourly rows:")
    print(hourly.head().to_string())

    print("\nMissing hourly values after the coverage rule:")
    print(hourly[["active_kwh"]].isna().sum())


if __name__ == "__main__":
    main()
