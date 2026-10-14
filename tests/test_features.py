"""
Tests for the feature pipeline.

What they protect:
  - No target leakage: no feature may depend on the sale price of its own row, and test-set
    sale prices must never influence any feature.
  - Neighborhood statistics are exactly what the docs say (checked against independent,
    hand-written formulas, both with and without shrinkage toward the county).
  - Train/serve parity: the dashboard must build the same features the model was trained on.
"""

import numpy as np
import pandas as pd
import pytest

import features
from conftest import make_frames
from real_estate_config import FEATURE_COLS

NEIGHBORHOOD = features.NEIGHBORHOOD_COLS
PRICE_DERIVED = {'lastSoldPrice', 'listPrice', 'price_per_sqft', 'list_to_sold_ratio'}


def expected_stats(zip_prices, county_prices, k, global_stats):
    """Independent re-implementation of the documented neighborhood statistics"""
    n = len(zip_prices)
    gm, gs = global_stats['median'], global_stats['std']

    def std_or(prices, fallback):
        return float(np.std(prices, ddof=1)) if len(prices) > 1 else fallback

    if k == 0:
        if n == 0:
            return gm, gs, 0
        return float(np.median(zip_prices)), std_or(zip_prices, gs), n
    prior_median = float(np.median(county_prices)) if len(county_prices) else gm
    prior_std = std_or(county_prices, gs)
    if n == 0:
        return prior_median, prior_std, 0
    w = n / (n + k)
    return (w * float(np.median(zip_prices)) + (1 - w) * prior_median,
            w * std_or(zip_prices, prior_std) + (1 - w) * prior_std, n)


# ----------------------------------------------------------------------------
# Leakage guards (run with and without shrinkage)
# ----------------------------------------------------------------------------

def test_feature_list_has_no_price_columns():
    """Guards against re-adding features computed from the sale price"""
    assert PRICE_DERIVED.isdisjoint(FEATURE_COLS)
    assert not any('price_per' in c or 'sold' in c for c in FEATURE_COLS)


def test_changing_a_test_sale_price_changes_no_feature(split_frames):
    """Test-set prices must not leak into any row's features (train or test)"""
    k = split_frames['k']
    X_train, _ = features.make_xy(split_frames['train'])
    X_test, _ = features.make_xy(split_frames['test'])

    tampered = split_frames['test_df'].copy()
    tampered['lastSoldPrice'] = tampered['lastSoldPrice'] * 10
    tr2, te2, _, _ = features.add_neighborhood_features(split_frames['train_df'].copy(), tampered, k)

    pd.testing.assert_frame_equal(features.make_xy(tr2)[0], X_train)
    pd.testing.assert_frame_equal(features.make_xy(te2)[0], X_test)


def test_a_training_row_never_sees_its_own_price(split_frames):
    """Leave-one-out: changing a row's own price must not change that row's features"""
    k, train_df, baseline = split_frames['k'], split_frames['train_df'], split_frames['train']

    counts = train_df['zip'].value_counts()
    zip_code = counts[counts >= 3].index[0]
    own, other = train_df.index[train_df['zip'] == zip_code][:2]

    tampered = train_df.copy()
    tampered.loc[own, 'lastSoldPrice'] = tampered.loc[own, 'lastSoldPrice'] * 10
    tr2, _, _, _ = features.add_neighborhood_features(tampered, split_frames['test_df'].copy(), k)

    pd.testing.assert_series_equal(tr2.loc[own, NEIGHBORHOOD], baseline.loc[own, NEIGHBORHOOD])
    assert not np.allclose(tr2.loc[other, NEIGHBORHOOD].to_numpy(float),
                           baseline.loc[other, NEIGHBORHOOD].to_numpy(float)), "test would be vacuous"


def test_a_training_row_never_sees_its_own_price_via_the_county(two_county_df):
    """With shrinkage the county prior must also exclude the row's own sale"""
    frames = make_frames(two_county_df, 20)
    train_df = frames['train_df']
    own = train_df.index[0]
    tampered = train_df.copy()
    tampered.loc[own, 'lastSoldPrice'] = tampered.loc[own, 'lastSoldPrice'] * 50
    tr2, _, _, _ = features.add_neighborhood_features(tampered, frames['test_df'].copy(), 20)
    pd.testing.assert_series_equal(tr2.loc[own, NEIGHBORHOOD], frames['train'].loc[own, NEIGHBORHOOD])


def test_split_comes_before_price_stats(prepared_df):
    """Train and test are disjoint, roughly 80/20, and the lookup counts come from training sales only"""
    out = features.build_datasets(prepared_df)
    assert set(out['X_train'].index).isdisjoint(out['X_test'].index)
    assert len(out['X_train']) + len(out['X_test']) == len(prepared_df)
    assert 0.15 < len(out['X_test']) / len(prepared_df) < 0.25
    for z, n in out['train_df'].groupby('zip').size().items():
        assert out['zip_table'].loc[z, 'count'] == n


# ----------------------------------------------------------------------------
# Neighborhood statistics match the documented formulas
# ----------------------------------------------------------------------------

def test_training_rows_use_leave_one_out_stats(split_frames):
    train_df, train, gs, k = (split_frames[x] for x in ('train_df', 'train', 'global_stats', 'k'))
    for idx in train_df.index[:25]:
        row = train_df.loc[idx]
        others = train_df.drop(index=idx)
        median, std, count = expected_stats(
            others[others['zip'] == row['zip']]['lastSoldPrice'].to_numpy(),
            others['lastSoldPrice'].to_numpy(), k, gs)
        got = train.loc[idx]
        assert got['neighborhood_sales_count'] == count
        assert got['neighborhood_median_price'] == pytest.approx(median)
        assert got['neighborhood_price_std'] == pytest.approx(std)


def test_test_rows_use_all_training_sales(split_frames):
    train_df, test_df, test, gs, k = (split_frames[x] for x in ('train_df', 'test_df', 'test', 'global_stats', 'k'))
    for idx in test_df.index:
        zip_prices = train_df[train_df['zip'] == test_df.loc[idx, 'zip']]['lastSoldPrice'].to_numpy()
        median, std, count = expected_stats(zip_prices, train_df['lastSoldPrice'].to_numpy(), k, gs)
        got = test.loc[idx]
        assert got['neighborhood_sales_count'] == count
        assert got['neighborhood_median_price'] == pytest.approx(median)
        assert got['neighborhood_price_std'] == pytest.approx(std)


def test_shrinkage_uses_each_rows_own_county(two_county_df):
    """The prior is the row's COUNTY, not the whole dataset"""
    frames = make_frames(two_county_df, 20)
    train_df, train, gs = frames['train_df'], frames['train'], frames['global_stats']
    for idx in train_df.index[:25]:
        row = train_df.loc[idx]
        others = train_df.drop(index=idx)
        in_county = others[others['county_name'] == row['county_name']]
        median, std, _ = expected_stats(
            others[others['zip'] == row['zip']]['lastSoldPrice'].to_numpy(),
            in_county['lastSoldPrice'].to_numpy(), 20, gs)
        assert train.loc[idx, 'neighborhood_median_price'] == pytest.approx(median)
        assert train.loc[idx, 'neighborhood_price_std'] == pytest.approx(std)


def test_shrinkage_pulls_thin_zips_toward_the_county_more_than_thick_ones(prepared_df):
    """Weight on the ZIP's own sales is n/(n+k): more sales, less pull toward the county"""
    train_df, test_df = features.split_data(prepared_df)
    _, _, raw_table, _ = features.add_neighborhood_features(train_df.copy(), test_df.copy(), 0)
    _, _, shrunk_table, _ = features.add_neighborhood_features(train_df.copy(), test_df.copy(), 20)
    county_median = train_df['lastSoldPrice'].median()
    pull = ((shrunk_table['median'] - raw_table['median']).abs() /
            (raw_table['median'] - county_median).abs().replace(0, np.nan))
    thin, thick = shrunk_table['count'].idxmin(), shrunk_table['count'].idxmax()
    assert shrunk_table.loc[thin, 'count'] < shrunk_table.loc[thick, 'count']
    assert pull[thin] >= pull[thick]
    assert (pull.dropna() <= 1 + 1e-9).all()  # never overshoots the county value


def test_unseen_zip_falls_back_to_the_prior(split_frames):
    train_df, test_df, gs, k = (split_frames[x] for x in ('train_df', 'test_df', 'global_stats', 'k'))
    stranger = test_df.iloc[[0]].copy()
    stranger['zip'] = '99999'
    _, te, _, _ = features.add_neighborhood_features(train_df.copy(), stranger, k)
    median, std, count = expected_stats(np.array([]), train_df['lastSoldPrice'].to_numpy(), k, gs)
    assert te.iloc[0]['neighborhood_sales_count'] == 0 == count
    assert te.iloc[0]['neighborhood_median_price'] == pytest.approx(median)
    assert te.iloc[0]['neighborhood_price_std'] == pytest.approx(std)


def test_zip_table_has_what_the_dashboard_needs(split_frames):
    table = split_frames['zip_table']
    assert {'lat', 'lon', 'county', 'median', 'std', 'count', 'raw_median'} <= set(table.columns)
    assert not table[['lat', 'lon', 'median', 'std', 'count']].isna().any().any()
    seen = table[table['count'] > 0]
    assert not seen['raw_median'].isna().any()


# ----------------------------------------------------------------------------
# Flood proxy percentile and ZIP lookup
# ----------------------------------------------------------------------------

def test_flood_percentile_matches_pandas_rank(prepared_df):
    proxy = prepared_df['flood_risk_proxy'].to_numpy()
    expected = pd.Series(proxy).rank(pct=True).to_numpy()
    np.testing.assert_allclose(features.flood_percentile(proxy, np.sort(proxy)), expected)


def test_flood_proxy_is_zero_north_of_cutoff():
    assert features.flood_proxy(26.1) == 0
    assert features.flood_proxy(25.3) == pytest.approx(0.5)


def test_nearest_zip_picks_the_closest_centroid():
    table = pd.DataFrame({'lat': [25.7, 26.1], 'lon': [-80.2, -80.1]}, index=['A', 'B'])
    assert features.nearest_zip(table, 25.71, -80.19) == 'A'
    assert features.nearest_zip(table, 26.05, -80.12) == 'B'


def test_nearest_zip_distance_is_in_miles():
    table = pd.DataFrame({'lat': [25.70, 26.10], 'lon': [-80.20, -80.10]}, index=['A', 'B'])
    zip_code, miles = features.nearest_zip_with_distance(table, 25.70, -80.20)
    assert zip_code == 'A' and miles == pytest.approx(0.0, abs=1e-6)
    _, miles = features.nearest_zip_with_distance(table, 25.80, -80.20)   # 0.10 deg latitude
    assert miles == pytest.approx(6.9, abs=0.1)


# ----------------------------------------------------------------------------
# Train/serve parity (the dashboard builds features the same way as the pipeline)
# ----------------------------------------------------------------------------

@pytest.mark.parametrize('k', [0, 20])
def test_dashboard_row_matches_pipeline_row(prepared_df, k):
    out = features.build_datasets(prepared_df, shrinkage_k=k)
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
