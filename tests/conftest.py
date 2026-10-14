import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import features  # noqa: E402


@pytest.fixture(scope='session')
def raw_df():
    """
    Small synthetic stand-in for the cleaned step-1 output: 160 sales over 10 ZIPs.
    Prices are all distinct so leave-one-out removal is unambiguous.
    """
    rng = np.random.default_rng(0)
    zips = [f'331{i:02d}' for i in range(10)]
    centers = {z: (25.60 + 0.07 * i, -80.40 + 0.03 * i) for i, z in enumerate(zips)}
    rows = []
    for i in range(160):
        z = zips[i % 10]
        lat, lon = centers[z]
        sqft = int(rng.integers(600, 4000))
        zip_premium = 1.0 + 0.1 * (i % 10)
        price = float(round(sqft * 300 * zip_premium * rng.lognormal(0, 0.25), 2)) + i * 0.01
        rows.append({
            'zip': z,
            'lat': lat + rng.normal(0, 0.005),
            'lon': lon + rng.normal(0, 0.005),
            'beds': int(rng.integers(1, 6)),
            'baths': int(rng.integers(1, 5)),
            'sqft': sqft,
            'yr_built': int(rng.integers(1950, 2024)),
            'sub_type': rng.choice(['condo', 'townhouse', None]),
            'lastSoldPrice': price,
        })
    return pd.DataFrame(rows)


@pytest.fixture()
def prepared_df(raw_df):
    """Row-level features added (nothing here depends on the sale price)"""
    df = features.engineer_property_features(raw_df.copy())
    df = features.engineer_geospatial_features(df)
    df = features.engineer_flood_risk(df)
    return df


@pytest.fixture()
def two_county_df(prepared_df):
    """Same data with a county column: the first five ZIPs in county 'A', the rest in 'B'"""
    df = prepared_df.copy()
    df['county_name'] = np.where(df['zip'].isin(sorted(df['zip'].unique())[:5]), 'A', 'B')
    return df


def make_frames(df, k):
    train_df, test_df = features.split_data(df)
    tr, te, zip_table, global_stats = features.add_neighborhood_features(train_df.copy(), test_df.copy(), k)
    return {'train_df': train_df, 'test_df': test_df, 'train': tr, 'test': te,
            'zip_table': zip_table, 'global_stats': global_stats, 'k': k}


@pytest.fixture(params=[0, 20], ids=['no-shrinkage', 'shrinkage-k20'])
def split_frames(request, prepared_df):
    """Train/test frames with neighborhood features, for both the legacy and the shrunk behavior"""
    return make_frames(prepared_df, request.param)
