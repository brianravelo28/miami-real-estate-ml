"""
Feature engineering shared by the training pipeline and the dashboard.

Having one implementation matters: if the app and the pipeline each built features
their own way, the model would be asked to predict on inputs it was never trained on.

Leakage rules:
  - No feature is derived from the sale price of the row it describes.
  - The train/test split happens BEFORE any price-based neighborhood statistic.
  - Training rows get leave-one-out ZIP stats (their own sale excluded); test rows and
    dashboard predictions use ZIP stats from training sales only.
"""

import logging

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split

from real_estate_config import (
    BRICKELL, DOWNTOWN_MIAMI, FEATURE_COLS, MIAMI_BEACH, RANDOM_STATE, REFERENCE_YEAR, TEST_SIZE,
)

logger = logging.getLogger(__name__)

FLOOD_LAT_CUTOFF = 25.8
JITTER_DEGREES = 0.005  # ~0.35 miles of Gaussian noise on ZIP-centroid coordinates
NEIGHBORHOOD_COLS = ['neighborhood_median_price', 'neighborhood_price_std', 'neighborhood_sales_count']


# ============================================================================
# Row-level features (no sale price involved)
# ============================================================================

def engineer_property_features(df):
    """sqft_log, property_age, and type dummies"""
    logger.info("\n--- Property Features ---")

    df['sqft_log'] = np.log1p(df['sqft'])
    df['property_age'] = REFERENCE_YEAR - df['yr_built']

    prop_type_col = 'sub_type' if 'sub_type' in df.columns else 'type'
    df['is_condo'] = (df[prop_type_col].str.lower() == 'condo').astype(int)
    df['is_townhouse'] = (df[prop_type_col].str.lower() == 'townhouse').astype(int)
    logger.info("✓ sqft_log, property_age, is_condo, is_townhouse")
    return df


def add_coordinate_jitter(df, scale=JITTER_DEGREES, seed=RANDOM_STATE):
    """
    Add small seeded Gaussian noise to lat/lon so sales in the same ZIP don't stack on the map.
    Seeded so re-running the pipeline reproduces the same coordinates (and therefore the same model).
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df['lat'] = df['lat'] + rng.normal(0, scale, size=len(df))
    df['lon'] = df['lon'] + rng.normal(0, scale, size=len(df))
    return df


def distance_features(lat, lon):
    """Geodesic miles from a point to the three Miami reference landmarks"""
    def miles(ref):
        try:
            return geodesic((lat, lon), ref).miles
        except Exception:
            return np.nan
    return {
        'dist_downtown': miles(DOWNTOWN_MIAMI),
        'dist_brickell': miles(BRICKELL),
        'dist_beach': miles(MIAMI_BEACH),
    }


def engineer_geospatial_features(df):
    """Distance-to-landmark columns for every row"""
    logger.info("\n--- Geospatial Features ---")
    dists = df.apply(lambda r: pd.Series(distance_features(r['lat'], r['lon'])), axis=1)
    for col in dists.columns:
        df[col] = dists[col]
    logger.info("✓ dist_downtown, dist_brickell, dist_beach")
    return df


def flood_proxy(lat):
    """Latitude-based flood proxy (lower latitude = higher risk); a proxy, not flood data"""
    return np.clip(FLOOD_LAT_CUTOFF - np.asarray(lat, dtype=float), 0, None)


def flood_percentile(proxy, reference_sorted):
    """
    Percentile rank of `proxy` within `reference_sorted` (ascending), using average
    rank for ties, i.e. the same definition as pandas rank(pct=True).
    """
    proxy = np.atleast_1d(np.asarray(proxy, dtype=float))
    ref = np.asarray(reference_sorted, dtype=float)
    below = np.searchsorted(ref, proxy, side='left')
    at_or_below = np.searchsorted(ref, proxy, side='right')
    return ((below + 1 + at_or_below) / 2) / len(ref)


def engineer_flood_risk(df):
    """Flood proxy and its percentile across the full dataset (no price involved)"""
    logger.info("\n--- Flood Risk ---")
    df['flood_risk_proxy'] = flood_proxy(df['lat'])
    ref = np.sort(df['flood_risk_proxy'].to_numpy(dtype=float))
    df['flood_risk_percentile'] = flood_percentile(df['flood_risk_proxy'].to_numpy(), ref)
    logger.info("✓ flood_risk_percentile (latitude-based proxy)")
    return df


# ============================================================================
# Split, then price-based neighborhood features
# ============================================================================

def split_data(df):
    """80/20 split, stratified by target quartile (done before any price-based stat)"""
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
        return pd.DataFrame(rows, index=df.index, columns=NEIGHBORHOOD_COLS)

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


def build_datasets(df):
    """
    Cleaned frame (step 1 output) -> everything step 2 saves.
    Returns a dict: X_train, y_train, X_test, y_test, zip_table, global_stats,
    flood_proxy_sorted, plus train_df/test_df (row-level frames, for inspection and tests).
    """
    df = engineer_property_features(df.copy())
    df = engineer_geospatial_features(df)
    df = engineer_flood_risk(df)

    train_df, test_df = split_data(df)
    train_df, test_df, zip_table, global_stats = add_neighborhood_features(train_df, test_df)

    X_train, y_train = make_xy(train_df)
    X_test, y_test = make_xy(test_df)
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'zip_table': zip_table, 'global_stats': global_stats,
        'flood_proxy_sorted': np.sort(df['flood_risk_proxy'].to_numpy(dtype=float)),
        'train_df': train_df, 'test_df': test_df,
    }


# ============================================================================
# Serving: build the same features for one new property (used by the dashboard)
# ============================================================================

def nearest_zip(zip_table, lat, lon):
    """ZIP whose centroid is closest to the point (longitude scaled by latitude)"""
    lon_scale = np.cos(np.radians(lat))
    dist = np.hypot(zip_table['lat'] - lat, (zip_table['lon'] - lon) * lon_scale)
    return dist.idxmin()


def build_feature_row(beds, baths, sqft, property_age, is_condo, is_townhouse,
                      lat, lon, zip_stats, flood_proxy_sorted):
    """
    One-row DataFrame in FEATURE_COLS order for a new property.
    `zip_stats` has 'median', 'std', 'count' (a row of the saved zip_table).
    """
    row = {
        'beds': beds, 'baths': baths, 'sqft': sqft, 'sqft_log': np.log1p(sqft),
        'property_age': property_age, 'is_condo': int(is_condo), 'is_townhouse': int(is_townhouse),
        'lat': lat, 'lon': lon,
        **distance_features(lat, lon),
        'neighborhood_median_price': zip_stats['median'],
        'neighborhood_price_std': zip_stats['std'],
        'neighborhood_sales_count': zip_stats['count'],
        'flood_risk_percentile': float(flood_percentile(flood_proxy(lat), flood_proxy_sorted)[0]),
    }
    return pd.DataFrame([row])[FEATURE_COLS]
