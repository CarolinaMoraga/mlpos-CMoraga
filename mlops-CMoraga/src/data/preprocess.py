
"""
This module cleans the raw diabetes dataset by handling missing values,
removing duplicates, fixing data types, and removing outliers.
It validates and outputs a processed dataset ready for machine learning.
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_PATH = BASE_DIR / "data" / "raw" / "diabetes_dataset.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "train_v1.csv"

# Diabetes dataset columns
FEATURES = [
    "age", "sex", "bmi", "bp",
    "s1", "s2", "s3", "s4", "s5", "s6"
]

TARGET = "target"

REQUIRED_COLUMNS = FEATURES + [TARGET]


def load_data(path: Path) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows containing missing values."""

    return df.dropna().copy()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataset."""

    return df.drop_duplicates().copy()


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all required columns to numeric values."""

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove extreme outliers using the IQR method.
    The target variable is excluded.
    """

    for col in FEATURES:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[
            (df[col] >= lower_bound) &
            (df[col] <= upper_bound)
        ]

    return df.copy()


def validate_data(df: pd.DataFrame) -> None:
    """Validate the processed diabetes dataset."""

    # 1. Check required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    assert not missing_columns, (
        f"Missing required columns: {missing_columns}"
    )

    # 2. Ensure all columns are numeric
    assert (
        df.select_dtypes(include=[np.number]).shape[1] == df.shape[1]
    ), "Non-numeric data detected!"

    # 3. Check missing values
    assert df.isnull().sum().sum() == 0, (
        "Critical missing values found!"
    )

    # 4. Check duplicates
    assert not df.duplicated().any(), (
        "Duplicate rows detected!"
    )

    # 5. Check dataset size
    assert len(df) > 0, "Dataset is empty!"

    # 6. Check target
    assert TARGET in df.columns, "Target column missing!"

    print("Data Validation Passed!")


def clean_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps and return the cleaned DataFrame.
    """

    # Create a copy to avoid modifying the original DataFrame
    cleaned_df = input_df.copy()

    # Check required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in cleaned_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Fix data types
    cleaned_df = fix_data_types(cleaned_df)

    # Handle missing values generated during conversion
    cleaned_df = handle_missing_values(cleaned_df)

    # Remove duplicate rows
    cleaned_df = remove_duplicates(cleaned_df)

    # Remove outliers
    cleaned_df = handle_outliers(cleaned_df)

    return cleaned_df


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV without the index."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    """Run the full preprocessing pipeline."""

    print("Loading raw data...")
    df = load_data(RAW_PATH)

    print(f"Shape before cleaning: {df.shape}")

    print("\nCleaning data...")
    df_cleaned = clean_data(df)

    print(f"Shape after cleaning: {df_cleaned.shape}")

    print("\nCleaning summary:")
    print(f"Rows removed: {len(df) - len(df_cleaned)}")
    print(
        f"Missing values: "
        f"{df_cleaned.isnull().sum().sum()}"
    )
    print(
        f"Duplicate rows: "
        f"{df_cleaned.duplicated().sum()}"
    )

    print("\nValidating processed data...")
    validate_data(df_cleaned)

    print("\nSaving processed data...")
    save_data(df_cleaned, PROCESSED_PATH)

    print(f"\nFinal shape: {df_cleaned.shape}")
    print(f"Output path: {PROCESSED_PATH}")
    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    main()
