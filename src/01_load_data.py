"""
Step 1: Load and Validate Data
Loads the Florida sales CSV, keeps Miami-Dade + Broward (by geocoded county), cleans it,
and saves the result. The rules live in data_prep.py so they can be tested.
"""

import logging
import os
import pickle

import pandas as pd

from real_estate_config import *
import data_prep

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_data():
    """Load Florida Real Estate Dataset 2026 from Kaggle"""
    if not os.path.exists(RAW_DATA_PATH):
        logger.info(f"Dataset not found at {RAW_DATA_PATH}")
        logger.info("Download from: https://www.kaggle.com/datasets/kanchana1990/florida-real-estate-sold-dataset-2026")
        logger.info("Place CSV at: data/florida_sold_2026.csv")
        raise FileNotFoundError("Dataset not found. Please download manually.")

    logger.info(f"Loading dataset from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Loaded {len(df):,} records")
    return df


def log_diagnostics(df):
    """Print missing-value and price summaries before cleaning"""
    logger.info("\n=== Data Validation ===")
    cols = [c for c in data_prep.REQUIRED_COLS if c in df.columns or c == 'yr_built']
    logger.info("\nMissing values:")
    for col in cols:
        source = 'year_built' if col == 'yr_built' else col
        if source in df.columns:
            logger.info(f"  {col}: {df[source].isna().mean() * 100:.2f}%")
    logger.info("\nPrice statistics:")
    logger.info(f"  Min: ${df['lastSoldPrice'].min():,.0f}")
    logger.info(f"  Max: ${df['lastSoldPrice'].max():,.0f}")
    logger.info(f"  Median: ${df['lastSoldPrice'].median():,.0f}")
    logger.info(f"  Mean: ${df['lastSoldPrice'].mean():,.0f}")


def explore_data(df):
    """Print exploratory statistics"""
    logger.info("\n=== Data Exploration ===")
    logger.info(f"\nData shape: {df.shape}")
    logger.info("\nBedrooms distribution:")
    logger.info(df['beds'].value_counts().sort_index().head(10))
    logger.info("\nGeographic bounds:")
    logger.info(f"  Latitude: {df['lat'].min():.4f} to {df['lat'].max():.4f}")
    logger.info(f"  Longitude: {df['lon'].min():.4f} to {df['lon'].max():.4f}")
    logger.info("\nSquare footage:")
    logger.info(f"  Min: {df['sqft'].min():,.0f}")
    logger.info(f"  Max: {df['sqft'].max():,.0f}")
    logger.info(f"  Median: {df['sqft'].median():,.0f}")


def main():
    logger.info("=" * 60)
    logger.info("STEP 1: LOAD AND VALIDATE DATA")
    logger.info("=" * 60 + "\n")

    raw = load_data()

    logger.info("Geocoding ZIP codes (pgeocode)...")
    metro = data_prep.filter_miami_metro(raw)
    logger.info(f"Filtered to {', '.join(TARGET_COUNTIES)}: {len(metro):,} records ({len(metro) / len(raw) * 100:.1f}%)")

    log_diagnostics(metro)
    before = len(metro)
    df = data_prep.clean_sales(metro)
    logger.info(f"\n✓ Removed {before - len(df)} rows (price/size outliers or missing critical values)")
    logger.info(f"\nFinal dataset: {len(df):,} records")

    explore_data(df)

    with open(FEATURES_PATH, 'wb') as f:
        pickle.dump(df, f)
    logger.info(f"\n✓ Saved cleaned data to {FEATURES_PATH}")

    logger.info("\n" + "=" * 60)
    logger.info("✓ Step 1 Complete. Ready for feature engineering.")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
