"""
Step 1: Load and Validate Data
Downloads (or loads) Florida Real Estate Dataset from Kaggle
Filters to Miami metro, validates, saves to pickle
"""

import pandas as pd
import numpy as np
import pickle
import logging
import pgeocode
from real_estate_config import *

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load Florida Real Estate Dataset 2026 from Kaggle"""
    
    # Check if file exists
    if not os.path.exists(RAW_DATA_PATH):
        logger.info(f"Dataset not found at {RAW_DATA_PATH}")
        logger.info("Download from: https://www.kaggle.com/datasets/kanchana1990/florida-real-estate-sold-dataset-2026")
        logger.info("Place CSV at: data/florida_sold_2026.csv")
        raise FileNotFoundError("Dataset not found. Please download manually.")
    
    logger.info(f"Loading dataset from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Loaded {len(df):,} records")
    
    return df

def filter_miami_metro(df):
    """Filter to Miami-Dade + Broward counties using real ZIP-code geocoding"""

    # Ensure zip is a clean 5-digit string (raw column is float64, e.g. 33446.0)
    df['zip'] = pd.to_numeric(df['zip'], errors='coerce').astype('Int64').astype(str).str.zfill(5)

    # Geocode every unique ZIP once (real lat/lon/county from pgeocode, not a hardcoded lookup)
    logger.info("Geocoding ZIP codes (pgeocode)...")
    nomi = pgeocode.Nominatim('us')
    unique_zips = df['zip'].unique().tolist()
    geo = nomi.query_postal_code(unique_zips).set_index('postal_code')

    df['county_name'] = df['zip'].map(geo['county_name'])
    df['lat'] = df['zip'].map(geo['latitude'])
    df['lon'] = df['zip'].map(geo['longitude'])

    # Filter by actual county, not ZIP prefix (prefixes bleed into Palm Beach/Monroe counties)
    metro_mask = df['county_name'].isin(TARGET_COUNTIES)
    df_metro = df[metro_mask].copy()

    # Sanity bounding box: catches occasional bad geocodes in the ZIP database
    # (e.g. ZIP 33973 is mislabeled "Broward" but geocodes to Lehigh Acres/Lee County)
    bounds_mask = (
        df_metro['lat'].between(*METRO_LAT_BOUNDS) &
        df_metro['lon'].between(*METRO_LON_BOUNDS)
    )
    dropped = (~bounds_mask).sum()
    if dropped:
        logger.info(f"  Dropped {dropped} record(s) with out-of-region coordinates (bad ZIP geocode)")
    df_metro = df_metro[bounds_mask].copy()

    logger.info(f"Filtered to {', '.join(TARGET_COUNTIES)}: {len(df_metro):,} records ({len(df_metro)/len(df)*100:.1f}%)")

    return df_metro

def validate_data(df):
    """Basic validation and cleaning"""

    logger.info("\n=== Data Validation ===")

    # Rename year_built to yr_built if needed
    if 'year_built' in df.columns and 'yr_built' not in df.columns:
        df = df.rename(columns={'year_built': 'yr_built'})

    # lat/lon/county already assigned in filter_miami_metro via real ZIP geocoding.
    # Add small jitter so multiple sales in the same ZIP don't stack exactly on the map.
    df['lat'] = df['lat'] + np.random.normal(0, 0.005, size=len(df))
    df['lon'] = df['lon'] + np.random.normal(0, 0.005, size=len(df))

    # Add flood_risk if missing (default to 'X' = no risk)
    if 'flood_risk' not in df.columns:
        df['flood_risk'] = 'X'

    # Check required columns
    required_cols = ['zip', 'beds', 'baths', 'sqft', 'yr_built', 'lastSoldPrice', 'lat', 'lon']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"✓ All required columns present")

    # Missing values
    logger.info(f"\nMissing values:")
    for col in required_cols:
        missing_pct = df[col].isna().sum() / len(df) * 100
        logger.info(f"  {col}: {missing_pct:.2f}%")

    # Price range
    logger.info(f"\nPrice statistics:")
    logger.info(f"  Min: ${df['lastSoldPrice'].min():,.0f}")
    logger.info(f"  Max: ${df['lastSoldPrice'].max():,.0f}")
    logger.info(f"  Median: ${df['lastSoldPrice'].median():,.0f}")
    logger.info(f"  Mean: ${df['lastSoldPrice'].mean():,.0f}")

    # Remove outliers
    initial_count = len(df)
    df = df[(df['lastSoldPrice'] >= MIN_PRICE) & (df['lastSoldPrice'] <= MAX_PRICE)].copy()
    df = df[(df['sqft'] >= MIN_SQFT) & (df['sqft'] <= MAX_SQFT)].copy()
    removed = initial_count - len(df)

    if removed > 0:
        logger.info(f"\n✓ Removed {removed} outliers (price or sqft)")

    # Remove rows with missing critical values
    critical_cols = ['beds', 'baths', 'sqft', 'yr_built', 'lastSoldPrice', 'lat', 'lon']
    initial_count = len(df)
    df = df.dropna(subset=critical_cols)
    removed = initial_count - len(df)

    if removed > 0:
        logger.info(f"✓ Removed {removed} rows with missing critical values")

    logger.info(f"\nFinal dataset: {len(df):,} records")

    return df

def explore_data(df):
    """Print exploratory statistics"""
    
    logger.info("\n=== Data Exploration ===")
    logger.info(f"\nData shape: {df.shape}")
    logger.info(f"\nColumn data types:")
    logger.info(df.dtypes)
    
    logger.info(f"\nBedrooms distribution:")
    logger.info(df['beds'].value_counts().sort_index().head(10))
    
    logger.info(f"\nProperty types:")
    if 'propertyType' in df.columns:
        logger.info(df['propertyType'].value_counts())
    
    logger.info(f"\nGeographic bounds:")
    logger.info(f"  Latitude: {df['lat'].min():.4f} to {df['lat'].max():.4f}")
    logger.info(f"  Longitude: {df['lon'].min():.4f} to {df['lon'].max():.4f}")
    
    logger.info(f"\nSquare footage:")
    logger.info(f"  Min: {df['sqft'].min():,.0f}")
    logger.info(f"  Max: {df['sqft'].max():,.0f}")
    logger.info(f"  Median: {df['sqft'].median():,.0f}")

def save_data(df):
    """Save cleaned data to pickle"""
    
    pickle.dump(df, open(FEATURES_PATH, 'wb'))
    logger.info(f"\n✓ Saved cleaned data to {FEATURES_PATH}")

def main():
    logger.info("="*60)
    logger.info("STEP 1: LOAD AND VALIDATE DATA")
    logger.info("="*60 + "\n")
    
    # Load
    df = load_data()
    
    # Filter to Miami metro
    df = filter_miami_metro(df)
    
    # Validate and clean
    df = validate_data(df)
    
    # Explore
    explore_data(df)
    
    # Save
    save_data(df)
    
    logger.info("\n" + "="*60)
    logger.info("✓ Step 1 Complete. Ready for feature engineering.")
    logger.info("="*60)

if __name__ == '__main__':
    main()
