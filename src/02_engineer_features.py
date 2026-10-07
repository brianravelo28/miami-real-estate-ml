"""
Step 2: Feature Engineering
Creates property, geospatial, neighborhood, and flood-proxy features.

Leakage rules:
  - No feature is derived from the sale price of the row it describes.
  - The train/test split happens BEFORE any price-based neighborhood statistic is computed.
  - Training rows get leave-one-out ZIP stats (their own sale is excluded); test rows and
    dashboard predictions use ZIP stats from training sales only.
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
    """Create property-level features (none use the sale price)"""
    logger.info("\n--- Property Features ---")

    df['sqft_log'] = np.log1p(df['sqft'])
    logger.info("✓ sqft_log")

    df['property_age'] = REFERENCE_YEAR - df['yr_built']
    logger.info("✓ property_age")

    # Property type dummies (use 'sub_type' if available, else 'type')
    prop_type_col = 'sub_type' if 'sub_type' in df.columns else 'type'
    df['is_condo'] = (df[prop_type_col].str.lower() == 'condo').astype(int)
    df['is_townhouse'] = (df[prop_type_col].str.lower() == 'townhouse').astype(int)
    logger.info("✓ is_condo, is_townhouse")

    return df

def engineer_geospatial_features(df):
    """Create location-based features"""
    logger.info("\n--- Geospatial Features ---")

    def distance_to(lat, lon, ref_point):
        try:
            return geodesic((lat, lon), ref_point).miles
        except Exception:
            return np.nan

    df['dist_downtown'] = df.apply(lambda r: distance_to(r['lat'], r['lon'], DOWNTOWN_MIAMI), axis=1)
    df['dist_brickell'] = df.apply(lambda r: distance_to(r['lat'], r['lon'], BRICKELL), axis=1)
    df['dist_beach'] = df.apply(lambda r: distance_to(r['lat'], r['lon'], MIAMI_BEACH), axis=1)
    logger.info("✓ dist_downtown, dist_brickell, dist_beach")

    return df

def flood_proxy(lat):
    """Latitude-based flood proxy (lower latitude = higher risk)"""
    return np.clip(25.8 - np.asarray(lat, dtype=float), 0, None)

def engineer_flood_risk(df):
    """Add flood risk proxy percentile (no price information involved)"""
    logger.info("\n--- Flood Risk ---")

    df['flood_risk_proxy'] = flood_proxy(df['lat'])
    df['flood_risk_percentile'] = df['flood_risk_proxy'].rank(pct=True)
    logger.info("✓ flood_risk_percentile (latitude-based proxy)")
    logger.info("  Note: Using latitude proxy. For production, use FEMA shapefiles.")

    return df

def split_data(df):
    """80/20 split, stratified by target quartile (done before any price-based stats)"""
    logger.info("\n--- Train/Test Split ---")

    y = np.log1p(df['lastSoldPrice'])
    strata = pd.qcut(y, 4, labels=False, duplicates='drop')

    train_idx, test_idx = train_test_split(
        df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strata
    )
    logger.info(f"✓ Train set: {len(train_idx):,} records")
    logger.info(f"✓ Test set: {len(test_idx):,} records")
    return df.loc[train_idx].copy(), df.loc[test_idx].copy()

def add_neighborhood_features(train_df, test_df):
    """
    ZIP-level price stats built from TRAINING sales only.
      - test rows: stats over all training sales in their ZIP
      - training rows: leave-one-out (their own sale is excluded)
    ZIPs with no (other) training sales fall back to global training stats.
    """
    logger.info("\n--- Neighborhood Features (train-only) ---")

    prices_by_zip = {z: g['lastSoldPrice'].to_numpy(dtype=float)
                     for z, g in train_df.groupby('zip')}
    all_train_prices = train_df['lastSoldPrice'].to_numpy(dtype=float)
    global_stats = {
        'median': float(np.median(all_train_prices)),
        'std': float(np.std(all_train_prices, ddof=1)),
        'count': 0.0,
    }

    def stats_from(prices):
        if len(prices) == 0:
            return global_stats['median'], global_stats['std'], 0.0
        std = float(np.std(prices, ddof=1)) if len(prices) > 1 else global_stats['std']
        return float(np.median(prices)), std, float(len(prices))

    def build(df, leave_one_out):
        rows = []
        for _, r in df.iterrows():
            prices = prices_by_zip.get(r['zip'], np.array([]))
            if leave_one_out and len(prices) > 0:
                # drop exactly one occurrence of this row's own sale price
                i = np.where(prices == float(r['lastSoldPrice']))[0]
                prices = np.delete(prices, i[0]) if len(i) else prices
            rows.append(stats_from(prices))
        out = pd.DataFrame(rows, index=df.index, columns=[
            'neighborhood_median_price', 'neighborhood_price_std', 'neighborhood_sales_count'])
        return out

    train_df = train_df.join(build(train_df, leave_one_out=True))
    test_df = test_df.join(build(test_df, leave_one_out=False))
    logger.info("✓ neighborhood_median_price, neighborhood_price_std, neighborhood_sales_count")

    # Lookup table for the dashboard: full-train stats + ZIP centroids
    zip_table = train_df.groupby('zip')['lastSoldPrice'].agg(
        median='median', std='std', count='count')
    zip_table['std'] = zip_table['std'].fillna(global_stats['std'])
    centroids = pd.concat([train_df, test_df]).groupby('zip')[['lat', 'lon']].mean()
    zip_table = centroids.join(zip_table, how='left')
    zip_table['median'] = zip_table['median'].fillna(global_stats['median'])
    zip_table['std'] = zip_table['std'].fillna(global_stats['std'])
    zip_table['count'] = zip_table['count'].fillna(0.0)

    return train_df, test_df, zip_table, global_stats

def make_xy(df):
    """Select modeling columns and the log-price target"""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    X = df[FEATURE_COLS].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = np.log1p(df['lastSoldPrice'])
    return X, y

def save_artifacts(X_train, X_test, y_train, y_test, zip_table, global_stats, flood_ref):
    """Save train/test data, target, and dashboard lookups"""
    pickle.dump(X_train, open(TRAIN_DATA_PATH, 'wb'))
    pickle.dump(X_test, open(TEST_DATA_PATH, 'wb'))
    pickle.dump(y_train, open(TRAIN_TARGET_PATH, 'wb'))
    pickle.dump(y_test, open(TEST_TARGET_PATH, 'wb'))
    pickle.dump({
        'zip_table': zip_table,
        'global_stats': global_stats,
        'flood_proxy_sorted': flood_ref,
    }, open(FEATURE_ARTIFACTS_PATH, 'wb'))

    logger.info(f"\n✓ Saved X_train to {TRAIN_DATA_PATH}")
    logger.info(f"✓ Saved X_test to {TEST_DATA_PATH}")
    logger.info(f"✓ Saved y_train to {TRAIN_TARGET_PATH}")
    logger.info(f"✓ Saved y_test to {TEST_TARGET_PATH}")
    logger.info(f"✓ Saved ZIP lookup artifacts to {FEATURE_ARTIFACTS_PATH}")

def main():
    logger.info("="*60)
    logger.info("STEP 2: ENGINEER FEATURES")
    logger.info("="*60)

    df = load_cleaned_data()

    df = engineer_property_features(df)
    df = engineer_geospatial_features(df)
    df = engineer_flood_risk(df)

    # Split first, then compute anything that touches sale prices
    train_df, test_df = split_data(df)
    train_df, test_df, zip_table, global_stats = add_neighborhood_features(train_df, test_df)

    X_train, y_train = make_xy(train_df)
    X_test, y_test = make_xy(test_df)
    logger.info(f"\n✓ Feature matrix: {X_train.shape[1]} features")
    logger.info(f"  Features: {', '.join(FEATURE_COLS)}")

    flood_ref = np.sort(df['flood_risk_proxy'].to_numpy(dtype=float))
    save_artifacts(X_train, X_test, y_train, y_test, zip_table, global_stats, flood_ref)

    logger.info("\n" + "="*60)
    logger.info("✓ Step 2 Complete. Ready for model training.")
    logger.info("="*60)

if __name__ == '__main__':
    main()
