"""
This module loads the raw dataset and introduces controlled noise to simulate
real-world data issues for the data engineering pipeline.
"""

import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


def load_raw_data() -> pd.DataFrame:
    # Load the diabetes dataset
    diabetes = load_diabetes()

    # Create a pandas DataFrame for features
    df = pd.DataFrame(
        diabetes.data,
        columns=diabetes.feature_names
    )

    # Add the target column
    df['target'] = diabetes.target

    return df


def save_dataset(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to a CSV file, creating directories as needed.

    Args:
        df: DataFrame to save.
        path: Destination file path.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    """Run the data ingestion pipeline."""
    output_path = BASE_DIR / "data" / "raw" / "diabetes_dataset.csv"
    
    print("Loading dataset...")
    df = load_raw_data()

    print("Saving dataset...")
    save_dataset(df, output_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Output path: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
