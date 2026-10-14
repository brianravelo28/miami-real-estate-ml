"""
Regression tests for step 1 (data_prep.py). Each one guards against a bug the raw data
caused earlier. A fake geocoder replaces the real ZIP database, so these run offline.
"""

import numpy as np
import pandas as pd
import pytest

import data_prep

# ZIP -> (county, lat, lon). Mirrors the real cases we hit. Boca Raton (33431, Palm Beach) is
# deliberately INSIDE the metro bounding box: only the county filter can exclude it.
FAKE_GEO = pd.DataFrame(
    {
        'county_name': ['Miami-Dade', 'Broward', 'Palm Beach', 'Monroe', 'Broward'],
        'latitude': [25.78, 26.12, 26.35, 24.73, 26.60],
        'longitude': [-80.20, -80.14, -80.13, -81.04, -81.73],
    },
    index=pd.Index(['33101', '33301', '33431', '33050', '33973'], name='postal_code'),
)


def fake_geocoder(zips):
    return FAKE_GEO.reindex(zips)


def raw_frame(zips, **overrides):
    n = len(zips)
    df = pd.DataFrame({
        'zip': zips,
        'beds': [3.0] * n, 'baths': [2.0] * n, 'sqft': [1500.0] * n,
        'year_built': [1990.0] * n, 'lastSoldPrice': [500_000] * n, 'listPrice': [520_000.0] * n,
        'type': ['single_family'] * n, 'sub_type': [None] * n,
    })
    for col, value in overrides.items():
        df[col] = value
    return df


# --- ZIP handling -----------------------------------------------------------

def test_float_zips_become_five_digit_strings():
    """The raw column is float64 (33446.0); str() of that matched no ZIP and geocoding silently failed"""
    out = data_prep.normalize_zip(pd.Series([33101.0, 33301.0]))
    assert out.tolist() == ['33101', '33301']


def test_leading_zero_zips_are_padded():
    assert data_prep.normalize_zip(pd.Series([1234.0])).iloc[0] == '01234'


def test_missing_zip_stays_missing_and_matches_nothing():
    out = data_prep.normalize_zip(pd.Series([33101.0, np.nan]))
    assert pd.isna(out.iloc[1])
    kept = data_prep.filter_miami_metro(raw_frame([33101.0, np.nan]), geocoder=fake_geocoder)
    assert kept['zip'].tolist() == ['33101']


def test_geocoder_receives_clean_unique_zips_without_missing():
    seen = {}

    def spy(zips):
        seen['zips'] = zips
        return fake_geocoder(zips)

    data_prep.filter_miami_metro(raw_frame([33101.0, 33101.0, np.nan, 33301.0]), geocoder=spy)
    assert sorted(seen['zips']) == ['33101', '33301']


# --- Geography filter -------------------------------------------------------

def test_filter_uses_county_not_zip_prefix():
    """ZIP 33431 (Boca Raton) starts with 334 but is Palm Beach; the old prefix filter let it in"""
    kept = data_prep.filter_miami_metro(raw_frame([33101.0, 33301.0, 33431.0, 33050.0]), geocoder=fake_geocoder)
    assert sorted(kept['zip']) == ['33101', '33301']
    assert set(kept['county_name']) == {'Miami-Dade', 'Broward'}


def test_mislabeled_zip_far_from_the_region_is_dropped():
    """ZIP 33973 is labeled Broward by the database but sits on the Gulf coast"""
    kept = data_prep.filter_miami_metro(raw_frame([33301.0, 33973.0]), geocoder=fake_geocoder)
    assert kept['zip'].tolist() == ['33301']


def test_unknown_zip_is_dropped_not_defaulted_to_downtown():
    kept = data_prep.filter_miami_metro(raw_frame([99999.0, 33101.0]), geocoder=fake_geocoder)
    assert kept['zip'].tolist() == ['33101']


# --- Cleaning rules ---------------------------------------------------------

def metro_frame(**overrides):
    return data_prep.filter_miami_metro(raw_frame([33101.0] * 6, **overrides), geocoder=fake_geocoder)


def test_price_outliers_are_removed():
    df = metro_frame(lastSoldPrice=[200, 49_999, 50_000, 500_000, 10_000_000, 10_000_001])
    out = data_prep.clean_sales(df)
    assert sorted(out['lastSoldPrice']) == [50_000, 500_000, 10_000_000]


def test_size_outliers_are_removed():
    df = metro_frame(sqft=[399.0, 400.0, 1500.0, 10_000.0, 10_001.0, 120.0])
    out = data_prep.clean_sales(df)
    assert sorted(out['sqft']) == [400.0, 1500.0, 10_000.0]


def test_rows_missing_critical_fields_are_dropped():
    df = metro_frame(beds=[np.nan, 3, 3, 3, 3, 3], year_built=[1990, np.nan, 1990, 1990, 1990, 1990])
    assert len(data_prep.clean_sales(df)) == 4


def test_year_built_is_renamed_and_missing_columns_raise():
    out = data_prep.clean_sales(metro_frame())
    assert 'yr_built' in out.columns and 'year_built' not in out.columns
    with pytest.raises(ValueError):
        data_prep.clean_sales(metro_frame().drop(columns=['sqft']))


def test_coordinate_jitter_is_applied_once_and_is_reproducible():
    base = metro_frame()
    a, b = data_prep.clean_sales(base), data_prep.clean_sales(base)
    pd.testing.assert_frame_equal(a, b)
    centroid = FAKE_GEO.loc['33101', 'latitude']
    assert not np.allclose(a['lat'], centroid)             # jitter happened
    assert (a['lat'] - centroid).abs().max() < 0.05        # and didn't happen twice or at a wild scale
