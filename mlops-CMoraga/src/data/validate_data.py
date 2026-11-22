"""
This module validates the dataset using Pydantic models to ensure data structure
and types before preprocessing.
"""

from typing import Optional

import pandas as pd
from pydantic import BaseModel, ValidationError
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_PATH = BASE_DIR / "data" / "raw" / "diabetes_dataset.csv"


class Diabetes(BaseModel):
    """Pydantic model representing a single row of diabetes data."""

    age: float       # Age of the patient
    sex: float       # Sex of the patient
    bmi: float       # Body mass index
    bp: float        # Average blood pressure
    s1: float        # Total serum cholesterol
    s2: float        # Low-density lipoproteins
    s3: float        # High-density lipoproteins
    s4: float        # Total cholesterol / HDL
    s5: float        # Log of serum triglycerides level
    s6: float        # Blood sugar level
    target: float    # Diabetes disease progression


def validate_row(row: dict, index: int) -> bool:
    """Try to validate a single row against the Diabetes model.

    Args:
        row: A dictionary representing one row of the dataset.
        index: The row index, used for error messages.

    Returns:
        True if the row is valid, False otherwise.
    """
    try:
        Diabetes(**row)
        return True
    except ValidationError as e:
        print(f"Row {index} failed validation: {e}")
        return False


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate all rows in the dataset and print a summary.

    Args:
        df: The pandas DataFrame to validate.
    """
    total = len(df)
    valid = 0
    invalid = 0

    for index, row in enumerate(df.to_dict("records")):
        if validate_row(row, index):
            valid += 1
        else:
            invalid += 1

    print("\n--- Validation Summary ---")
    print(f"Total rows:   {total}")
    print(f"Valid rows:   {valid}")
    print(f"Invalid rows: {invalid}")


def main():
    """Load the raw dataset and run validation."""
    path = RAW_PATH
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    validate_dataset(df)


if __name__ == "__main__":
    main()
