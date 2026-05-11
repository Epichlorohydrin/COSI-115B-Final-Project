"""Placeholder preprocessing utilities.

This module should be expanded to convert raw conversation logs into (query, response)
pairs. For now it contains a small helper to load a CSV and verify columns.
"""
import pandas as pd


def validate_pairs_csv(path: str):
    df = pd.read_csv(path)
    required = {'query', 'response'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv')
    args = parser.parse_args()
    df = validate_pairs_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
