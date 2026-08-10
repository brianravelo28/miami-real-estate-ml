"""
Step 2: Feature Engineering
Creates all property, geospatial, neighborhood, and temporal features
Outputs: X (features), y (target), train/test split
"""

import pandas as pd
import numpy as np
import pickle
import logging
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from real_estate_config import *

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_cleaned_data():
    """Load output from step 1"""
    logger.info("Loading cleaned data...")
    df = pickle.load(open(FEATURES_PATH, 'rb'))
    logger.info(f"Loaded {len(df):,} records")
    return df

def engineer_property_features(df):
    """Create property-level features"""
    logger.info("\n--- Property Features ---")
    
    # Log-transform sqft (right-skewed)
    df['sqft_log'] = np.log1p(df['sqft'])
    logger.info("✓ sqft_log")
    
    # Property age
    df['property_age'] = REFERENCE_YEAR - df['yr_built']
    logger.info("✓ property_age")
    
    # Price per sqft
    df['price_per_sqft'] = df['lastSoldPrice'] / df['sqft']
    logger.info("✓ price_per_sqft")
    
    # List to sold ratio (negotiation indicator)
    df['list_to_sold_ratio'] = df['lastSoldPrice'] / df['listPrice']
    # Cap at reasonable bounds (avoid infinities from $0 listings)
    df['list_to_sold_ratio'] = df['list_to_sold_ratio'].clip(0.5, 1.5)
    logger.info("✓ list_to_sold_ratio")
    
    # Property type dummies (use 'sub_type' if available, else 'type')
    prop_type_col = 'sub_type' if 'sub_type' in df.columns else 'type'
    df['is_condo'] = (df[prop_type_col].str.lower() == 'condo').astype(int)
    df['is_townhouse'] = (df[prop_type_col].str.lower() == 'townhouse').astype(int)
    logger.info("✓ is_condo, is_townhouse")
    
    # Bin property age for non-linear effects
    df['property_age_binned'] = pd.cut(
        df['property_age'],
        bins=PROPERTY_AGE_BINS,
        labels=PROPERTY_AGE_LABELS,
        include_lowest=True
    )
    # Convert to ordinal (0, 1, 2, 3)
    df['property_age_ordinal'] = df['property_age_binned'].cat.codes
    logger.info("✓ property_age_ordinal")
    
    return df

def engineer_geospatial_features(df):
    """Create location-based features"""
    logger.info("\n--- Geospatial Features ---")
    
    # Distance calculations using haversine
    def haversine_distance(lat, lon, ref_point):
        try:
            return geodesic((lat, lon), ref_point).miles
        except:
            return np.nan
    
    # Distance to key locations
    df['dist_downtown'] = df.apply(
        lambda row: haversine_distance(row['lat'], row['lon'], DOWNTOWN_MIAMI),
        axis=1
    )
    logger.info("✓ dist_downtown")
    
    df['dist_brickell'] = df.apply(
        lambda row: haversine_distance(row['lat'], row['lon'], BRICKELL),
        axis=1
    )
    logger.info("✓ dist_brickell")
    
    df['dist_beach'] = df.apply(
        lambda row: haversine_distance(row['lat'], row['lon'], MIAMI_BEACH),
        axis=1
    )
    logger.info("✓ dist_beach")
    
    # Distance to coast (approximate: any point within 5 miles of ocean is waterfront-adjacent)
    # For MVP, use simple proxy: latitude > 25.8 (generally closer to coast)
    df['near_coast'] = (df['lat'] > 25.8).astype(int)
    logger.info("✓ near_coast (proxy)")
    
    return df

def engineer_neighborhood_features(df):
    """Create ZIP-level aggregated features"""
    logger.info("\n--- Neighborhood Features ---")
    
    # ZIP-level statistics
    zip_stats = df.groupby('zip')['lastSoldPrice'].agg([
        ('neighborhood_median_price', 'median'),
        ('neighborhood_price_std', 'std'),
        ('neighborhood_sales_count', 'count'),
    ]).reset_index()
    
    # Merge back
    df = df.merge(zip_stats, on='zip', how='left')
    logger.info("✓ neighborhood_median_price, neighborhood_price_std, neighborhood_sales_count")
    
    # Price tier (derived from median)
    def assign_price_tier(price):
        if pd.isna(price):
            return 1  # Default to mainstream
        elif price < 300_000:
            return 0  # Budget
        elif price < 500_000:
            return 1  # Mainstream
        elif price < 1_000_000:
            return 2  # Luxury
        else:
            return 3  # Ultra-luxury
    
    df['neighborhood_price_tier'] = df['neighborhood_median_price'].apply(assign_price_tier)
    logger.info("✓ neighborhood_price_tier")
    
    return df

def engineer_temporal_features(df):
    """Create time-based features"""
    logger.info("\n--- Temporal Features ---")
    
    # Extract year and month from sale date
    if 'saleDate' in df.columns:
        df['saleDate'] = pd.to_datetime(df['saleDate'], errors='coerce')
        df['sale_year'] = df['saleDate'].dt.year
        df['sale_month'] = df['saleDate'].dt.month
        df['sale_quarter'] = df['saleDate'].dt.quarter
        logger.info("✓ sale_year, sale_month, sale_quarter")
    else:
        # Fallback: use 2026 as default
        df['sale_year'] = 2026
        df['sale_month'] = 6  # Mid-year default
        df['sale_quarter'] = 2
        logger.info("✓ sale_year, sale_month, sale_quarter (defaults)")
    
    return df

def engineer_flood_risk(df):
    """Add flood risk (simplified version - ordinal encoding)"""
    logger.info("\n--- Flood Risk ---")
    
    # Simplified: use latitude as proxy for flood risk
    # Lower latitude (closer to coast) = higher risk
    df['flood_risk_proxy'] = (25.8 - df['lat']).clip(0, None)
    df['flood_risk_percentile'] = df['flood_risk_proxy'].rank(pct=True)
    logger.info("✓ flood_risk_percentile (latitude-based proxy)")
    
    # For production, this would be a proper FEMA spatial join
    logger.info("  Note: Using latitude proxy. For production, use FEMA shapefiles.")
    
    return df

def prepare_modeling_data(df):
    """Select features, handle nulls, encode categoricals"""
    logger.info("\n--- Prepare for Modeling ---")
    
    # Handle missing values
    df['neighborhood_price_std'] = df['neighborhood_price_std'].fillna(df['neighborhood_price_std'].median())
    df['list_to_sold_ratio'] = df['list_to_sold_ratio'].fillna(1.0)
    logger.info("✓ Filled missing values")
    
    # Select feature columns (from config)
    available_features = [col for col in FEATURE_COLS if col in df.columns]
    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    
    if missing_features:
        logger.info(f"⚠ Missing features (will skip): {missing_features}")
    
    X = df[available_features].copy()
    y = np.log1p(df['lastSoldPrice'])  # Log-transform target
    
    logger.info(f"✓ Feature matrix shape: {X.shape}")
    logger.info(f"  Features: {', '.join(available_features[:5])}... ({len(available_features)} total)")
    
    return X, y, df

def train_test_split_stratified(X, y, df_full):
    """Split data, stratified by neighborhood tier"""
    logger.info("\n--- Train/Test Split ---")

    # Use neighborhood_price_tier for stratification if available
    stratify_by = None
    if 'neighborhood_price_tier' in df_full.columns:
        stratify_by = df_full['neighborhood_price_tier']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_by
    )

    logger.info(f"✓ Train set: {len(X_train):,} records")
    logger.info(f"✓ Test set: {len(X_test):,} records")

    return X_train, X_test, y_train, y_test

def save_artifacts(X_train, X_test, y_train, y_test):
    """Save train/test data and target"""
    
    pickle.dump(X_train, open(TRAIN_DATA_PATH, 'wb'))
    pickle.dump(X_test, open(TEST_DATA_PATH, 'wb'))
    pickle.dump(y_train, open(TRAIN_TARGET_PATH, 'wb'))
    pickle.dump(y_test, open(TEST_TARGET_PATH, 'wb'))
    
    logger.info(f"\n✓ Saved X_train to {TRAIN_DATA_PATH}")
    logger.info(f"✓ Saved X_test to {TEST_DATA_PATH}")
    logger.info(f"✓ Saved y_train to {TRAIN_TARGET_PATH}")
    logger.info(f"✓ Saved y_test to {TEST_TARGET_PATH}")

def main():
    logger.info("="*60)
    logger.info("STEP 2: ENGINEER FEATURES")
    logger.info("="*60)
    
    # Load
    df = load_cleaned_data()
    
    # Engineer features in sequence
    df = engineer_property_features(df)
    df = engineer_geospatial_features(df)
    df = engineer_neighborhood_features(df)
    df = engineer_temporal_features(df)
    df = engineer_flood_risk(df)
    
    # Prepare for modeling
    X, y, df_full = prepare_modeling_data(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, df_full)
    
    # Save
    save_artifacts(X_train, X_test, y_train, y_test)
    
    logger.info("\n" + "="*60)
    logger.info("✓ Step 2 Complete. Ready for model training.")
    logger.info("="*60)

if __name__ == '__main__':
    main()
