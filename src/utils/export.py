# src/utils/export.py

"""
Export utilities for NewsBot 2.0
Exports reports, dataframes, and model results.
"""

import pandas as pd

def export_to_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"Saved DataFrame to {filename}")

def export_to_excel(df, filename):
    df.to_excel(filename, index=False)
    print(f"Saved DataFrame to {filename}")

def export_report(text, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved report to {filename}")
