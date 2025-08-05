# src/data_processing/data_validator.py

"""
Data Validator for NewsBot 2.0
Checks for missing values, required columns, and basic data quality.
"""

import pandas as pd

class DataValidator:
    def __init__(self, required_columns=None):
        if required_columns is None:
            required_columns = ['content', 'category']
        self.required_columns = required_columns

    def check_columns(self, df):
        """Ensures required columns exist in the DataFrame."""
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True

    def check_missing(self, df):
        """Checks for missing (null/empty) values in required columns."""
        for col in self.required_columns:
            if df[col].isnull().any() or (df[col].astype(str).str.strip() == '').any():
                raise ValueError(f"Column '{col}' contains missing or empty values.")
        return True

    def validate(self, df):
        """Runs all validation checks on the DataFrame."""
        self.check_columns(df)
        self.check_missing(df)
        return True
