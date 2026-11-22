"""
This module applies feature engineering to the cleaned Diabetes dataset.
It creates interaction features and serum measurement ratios, then saves
the resulting dataset ready for model training.
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data" / "processed" / "train_v1.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "train_features_v1.csv"

FEATURES = [
    "age", "sex", "bmi", "bp",
    "s1", "s2", "s3", "s4", "s5", "s6"
]

TARGET = "target"


def load_data(path: Path) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""
    return pd.read_csv(path)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create three engineered features:

    - bmi_age_interaction: interaction between BMI and age.
    - bp_bmi_interaction: interaction between blood pressure and BMI.
    - s1_s2_ratio: ratio between serum measurements s1 and s2.
    """

    df = df.copy()

    # Check required columns
    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # 1. Interaction between BMI and Age
    df["bmi_age_interaction"] = df["bmi"] * df["age"]

    # 2. Interaction between Blood Pressure and BMI
    df["bp_bmi_interaction"] = df["bp"] * df["bmi"]

    # 3. Ratio between serum measurements
    df["s1_s2_ratio"] = df["s1"] / (df["s2"] + 1e-9)

    # Check generated features for invalid values
    engineered_features = [
        "bmi_age_interaction",
        "bp_bmi_interaction",
        "s1_s2_ratio",
    ]

    if not np.isfinite(df[engineered_features].to_numpy()).all():
        raise ValueError(
            "Feature engineering generated NaN or infinite values."
        )

    return df


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV without the index."""

    if path.exists():
        raise FileExistsError(
            f"Output file already exists: {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    """Run the full feature engineering pipeline."""

    print("Loading processed data...")
    df = load_data(INPUT_PATH)

    print(f"Rows before feature engineering: {len(df)}")

    print("\nApplying feature engineering...")
    df = feature_engineering(df)

    print("\nNew features created:")
    print([
        "bmi_age_interaction",
        "bp_bmi_interaction",
        "s1_s2_ratio",
    ])

    print("\nSaving dataset...")
    save_data(df, OUTPUT_PATH)

    print(f"Final shape: {df.shape}")
    print(f"\nOutput path: {OUTPUT_PATH}")
    print("Feature engineering completed successfully!")


if __name__ == "__main__":
    main()