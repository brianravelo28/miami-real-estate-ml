import os

import pandas as pd
import pytest

import real_estate_config as cfg
import summaries


def sales():
    rows = []
    for z, n, price in [('33101', 6, 500_000), ('33102', 2, 900_000)]:
        for i in range(n):
            rows.append({'type': 'condos' if i % 2 else 'single_family', 'lastSoldPrice': price + i * 1000,
                         'sqft': 1000.0, 'yr_built': 1975.0, 'zip': z, 'county_name': 'Miami-Dade',
                         'lat': 25.8, 'lon': -80.2})
    return summaries.prepare(pd.DataFrame(rows))


def test_prepare_adds_dashboard_columns():
    df = sales()
    assert set(df['property_type']) == {'Condo', 'Single family'}
    assert (df['price_per_sqft'] == df['lastSoldPrice'] / df['sqft']).all()
    assert (df['build_period'].astype(str) == '1960-1979').all()


def test_unknown_types_fall_into_other():
    df = summaries.prepare(pd.DataFrame({'type': ['mobile', 'something_new'], 'lastSoldPrice': [1.0, 2.0],
                                         'sqft': [1.0, 1.0], 'yr_built': [2000.0, 2000.0]}))
    assert list(df['property_type']) == ['Other', 'Other']


def test_zip_summary_counts_medians_and_flags_thin_zips():
    table = summaries.zip_summary(sales()).set_index('zip')
    assert table.loc['33101', 'sales'] == 6 and table.loc['33102', 'sales'] == 2
    assert table.loc['33101', 'median_price'] == 502_500  # median of 500000..505000
    assert not table.loc['33101', 'thin'] and table.loc['33102', 'thin']


def test_thin_threshold_comes_from_config():
    assert summaries.zip_summary(sales(), min_sales=1)['thin'].sum() == 0
    assert summaries.zip_summary(sales(), min_sales=cfg.MIN_ZIP_SALES)['thin'].sum() == 1


def test_committed_dataset_matches_the_dashboard_contract():
    if not os.path.exists(cfg.CLEAN_DATA_PATH):
        pytest.skip("sales_clean.csv not available")
    df = pd.read_csv(cfg.CLEAN_DATA_PATH, dtype={'zip': str})
    assert set(cfg.OUTPUT_COLS) == set(df.columns)
    assert df['zip'].str.fullmatch(r'\d{5}').all()
    assert set(df['county_name']) <= set(cfg.TARGET_COUNTIES)
    assert df['lastSoldPrice'].between(cfg.MIN_PRICE, cfg.MAX_PRICE).all()
    assert df['sqft'].between(cfg.MIN_SQFT, cfg.MAX_SQFT).all()
    full = summaries.prepare(df)
    assert full['property_type'].isin(['Single family', 'Condo', 'Townhome', 'Other']).all()
    assert full['build_period'].notna().all()
