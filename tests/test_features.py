"""
Tests for the feature pipeline.

What they protect:
  - No target leakage: no feature may depend on the sale price of its own row, and test-set
    sale prices must never influence any feature.
  - Train/serve parity: the dashboard must build the same features the model was trained on.
"""

import numpy as np
import pandas as pd
import pytest

import features
from real_estate_config import FEATURE_COLS

NEIGHBORHOOD = features.NEIGHBORHOOD_COLS
PRICE_DERIVED = {'lastSoldPrice', 'listPrice', 'price_per_sqft', 'list_to_sold_ratio'}


# ----------------------------------------------------------------------------
# Leakage guards
# ----------------------------------------------------------------------------

def test_feature_list_has_no_price_columns():
    """Guards against re-adding features computed from the sale price"""
    assert PRICE_DERIVED.isdisjoint(FEATURE_COLS)
    assert not any('price_per' in c or 'sold' in c for c in FEATURE_COLS)


def test_changing_a_test_sale_price_changes_no_feature(split_frames):
    """Test-set prices must not leak into any row's features (train or test)"""
    train_df, test_df = split_frames['train_df'], split_frames['test_df']
    X_train, _ = features.make_xy(split_frames['train'])
    X_test, _ = features.make_xy(split_frames['test'])

    tampered = test_df.copy()
    tampered['lastSoldPrice'] = tampered['lastSoldPrice'] * 10
    tr2, te2, _, _ = features.add_neighborhood_features(train_df.copy(), tampered)

    pd.testing.assert_frame_equal(features.make_xy(tr2)[0], X_train)
    pd.testing.assert_frame_equal(features.make_xy(te2)[0], X_test)


def test_a_training_row_never_sees_its_own_price(split_frames):
    """Leave-one-out: changing a row's own price must not change that row's features"""
    train_df, test_df = split_frames['train_df'], split_frames['test_df']
    baseline = split_frames['train']

    counts = train_df['zip'].value_counts()
    zip_code = counts[counts >= 3].index[0]
    own, other = train_df.index[train_df['zip'] == zip_code][:2]

    tampered = train_df.copy()
    tampered.loc[own, 'lastSoldPrice'] = tampered.loc[own, 'lastSoldPrice'] * 10
    tr2, _, _, _ = features.add_neighborhood_features(tampered, test_df.copy())

    # its own features are unchanged...
    pd.testing.assert_series_equal(tr2.loc[own, NEIGHBORHOOD], baseline.loc[own, NEIGHBORHOOD])
    # ...but a neighbor's are (so this test isn't vacuous)
    assert not np.allclose(tr2.loc[other, NEIGHBORHOOD].to_numpy(float),
                           baseline.loc[other, NEIGHBORHOOD].to_numpy(float))


def test_split_comes_before_price_stats(prepared_df):
    """Train and test are disjoint, roughly 80/20, and test rows add nothing to training stats"""
    out = features.build_datasets(prepared_df)
    assert set(out['X_train'].index).isdisjoint(out['X_test'].index)
    assert len(out['X_train']) + len(out['X_test']) == len(prepared_df)
    assert 0.15 < len(out['X_test']) / len(prepared_df) < 0.25
    # the saved lookup table is built from training sales only
    train_counts = out['train_df'].groupby('zip').size()
    for z, n in train_counts.items():
        assert out['zip_table'].loc[z, 'count'] == n


# ----------------------------------------------------------------------------
# Neighborhood statistics are what we say they are
# ----------------------------------------------------------------------------

def test_training_rows_use_leave_one_out_stats(split_frames):
    train_df, train = split_frames['train_df'], split_frames['train']
    gs = split_frames['global_stats']
    for idx in train_df.index[:25]:
        zip_code = train_df.loc[idx, 'zip']
        others = train_df[(train_df['zip'] == zip_code) & (train_df.index != idx)]['lastSoldPrice']
        row = train.loc[idx]
        assert row['neighborhood_sales_count'] == len(others)
        if len(others) == 0:
            assert row['neighborhood_median_price'] == pytest.approx(gs['median'])
            continue
        assert row['neighborhood_median_price'] == pytest.approx(others.median())
        expected_std = others.std(ddof=1) if len(others) > 1 else gs['std']
        assert row['neighborhood_price_std'] == pytest.approx(expected_std)


def test_test_rows_use_all_training_sales_in_their_zip(split_frames):
    train_df, test_df, test = split_frames['train_df'], split_frames['test_df'], split_frames['test']
    for idx in test_df.index:
        zip_prices = train_df[train_df['zip'] == test_df.loc[idx, 'zip']]['lastSoldPrice']
        assert test.loc[idx, 'neighborhood_sales_count'] == len(zip_prices)
        assert test.loc[idx, 'neighborhood_median_price'] == pytest.approx(zip_prices.median())


def test_unseen_zip_falls_back_to_global_training_stats(split_frames):
    train_df, test_df = split_frames['train_df'], split_frames['test_df']
    stranger = test_df.iloc[[0]].copy()
    stranger['zip'] = '99999'
    _, te, _, gs = features.add_neighborhood_features(train_df.copy(), stranger)
    row = te.iloc[0]
    assert row['neighborhood_sales_count'] == 0
    assert row['neighborhood_median_price'] == pytest.approx(gs['median'])
    assert row['neighborhood_price_std'] == pytest.approx(gs['std'])


# ----------------------------------------------------------------------------
# Flood proxy percentile
# ----------------------------------------------------------------------------

def test_flood_percentile_matches_pandas_rank(prepared_df):
    proxy = prepared_df['flood_risk_proxy'].to_numpy()
    ref = np.sort(proxy)
    expected = pd.Series(proxy).rank(pct=True).to_numpy()
    np.testing.assert_allclose(features.flood_percentile(proxy, ref), expected)


def test_flood_proxy_is_zero_north_of_cutoff():
    assert features.flood_proxy(26.1) == 0
    assert features.flood_proxy(25.3) == pytest.approx(0.5)


# ----------------------------------------------------------------------------
# Train/serve parity (the dashboard builds features the same way as the pipeline)
# ----------------------------------------------------------------------------

def test_dashboard_row_matches_pipeline_row(prepared_df):
    out = features.build_datasets(prepared_df)
    test_df, X_test = out['test_df'], out['X_test']

    for idx in X_test.index[:10]:
        r = test_df.loc[idx]
        served = features.build_feature_row(
            beds=r['beds'], baths=r['baths'], sqft=r['sqft'], property_age=r['property_age'],
            is_condo=r['is_condo'], is_townhouse=r['is_townhouse'],
            lat=r['lat'], lon=r['lon'],
            zip_stats=out['zip_table'].loc[r['zip']],
            flood_proxy_sorted=out['flood_proxy_sorted'],
        )
        assert list(served.columns) == list(X_test.columns)
        np.testing.assert_allclose(served.iloc[0].to_numpy(float), X_test.loc[idx].to_numpy(float),
                                   rtol=1e-9, err_msg=f"row {idx} differs between app and pipeline")


def test_nearest_zip_picks_the_closest_centroid():
    table = pd.DataFrame({'lat': [25.7, 26.1], 'lon': [-80.2, -80.1]}, index=['A', 'B'])
    assert features.nearest_zip(table, 25.71, -80.19) == 'A'
    assert features.nearest_zip(table, 26.05, -80.12) == 'B'
