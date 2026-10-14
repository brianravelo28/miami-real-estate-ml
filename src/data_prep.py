"""
Data preparation for step 1: ZIP cleaning, geography filtering, and sales cleaning.

Kept separate from the script (01_load_data.py) so each rule can be tested without
downloading the Kaggle file or the ZIP-code database. Each rule here exists because
the raw data broke something earlier (see tests/test_data_prep.py).
"""

import logging

import numpy as np
import pandas as pd

import features
from real_estate_config import (
    MAX_PRICE, MAX_SQFT, METRO_LAT_BOUNDS, METRO_LON_BOUNDS, MIN_PRICE, MIN_SQFT, TARGET_COUNTIES,
)

logger = logging.getLogger(__name__)

REQUIRED_COLS = ['zip', 'beds', 'baths', 'sqft', 'yr_built', 'lastSoldPrice', 'lat', 'lon']
CRITICAL_COLS = ['beds', 'baths', 'sqft', 'yr_built', 'lastSoldPrice', 'lat', 'lon']


def normalize_zip(zip_series):
    """
    ZIPs arrive as floats (33446.0). Convert to zero-padded 5-character strings;
    missing ZIPs stay missing so they can't accidentally match a real code.
    """
    numeric = pd.to_numeric(zip_series, errors='coerce')
    as_text = numeric.astype('Int64').astype(str).str.zfill(5)
    return as_text.where(numeric.notna(), np.nan)


def geocode_zips(unique_zips):
    """ZIP -> (county_name, latitude, longitude) using the offline pgeocode database"""
    import pgeocode  # imported here so the rest of this module is usable without it
    table = pgeocode.Nominatim('us').query_postal_code(list(unique_zips))
    return table.set_index('postal_code')[['county_name', 'latitude', 'longitude']]


def attach_geography(df, geo):
    """Add county_name, lat, lon from a table indexed by 5-digit ZIP"""
    df = df.copy()
    df['county_name'] = df['zip'].map(geo['county_name'])
    df['lat'] = df['zip'].map(geo['latitude'])
    df['lon'] = df['zip'].map(geo['longitude'])
    return df


def filter_to_target_area(df):
    """
    Keep rows in the target counties whose coordinates are plausible for them.
      - County, not ZIP prefix: prefixes like 334 also cover Palm Beach and the Keys.
      - Bounding box: the ZIP database occasionally pairs a target county name with
        coordinates far away (ZIP 33973 is labeled Broward but is on the Gulf coast).
    """
    in_county = df[df['county_name'].isin(TARGET_COUNTIES)]
    in_box = (in_county['lat'].between(*METRO_LAT_BOUNDS) &
              in_county['lon'].between(*METRO_LON_BOUNDS))
    dropped = int((~in_box).sum())
    if dropped:
        logger.info(f"  Dropped {dropped} record(s) with out-of-region coordinates (bad ZIP geocode)")
    return in_county[in_box].copy()


def filter_miami_metro(df, geocoder=geocode_zips):
    """Raw frame -> rows in Miami-Dade + Broward, with ZIP, county, lat, lon attached"""
    df = df.copy()
    df['zip'] = normalize_zip(df['zip'])
    geo = geocoder(df['zip'].dropna().unique().tolist())
    df = attach_geography(df, geo)
    return filter_to_target_area(df)


def clean_sales(df):
    """Rename, jitter coordinates, drop price/size outliers and rows missing critical fields"""
    if 'year_built' in df.columns and 'yr_built' not in df.columns:
        df = df.rename(columns={'year_built': 'yr_built'})

    # Seeded jitter so same-ZIP sales don't stack exactly on the map
    df = features.add_coordinate_jitter(df)

    if 'flood_risk' not in df.columns:
        df['flood_risk'] = 'X'

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[(df['lastSoldPrice'] >= MIN_PRICE) & (df['lastSoldPrice'] <= MAX_PRICE)].copy()
    df = df[(df['sqft'] >= MIN_SQFT) & (df['sqft'] <= MAX_SQFT)].copy()
    return df.dropna(subset=CRITICAL_COLS)
