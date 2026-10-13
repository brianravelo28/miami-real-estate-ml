"""
Checks on the saved model and data files that the deployed app loads.
These skip automatically if a file isn't present (for example, a fresh clone without Git LFS).
"""

import os
import pickle

import numpy as np
import pandas as pd
import pytest

import features
from real_estate_config import (
    FEATURE_ARTIFACTS_PATH, FEATURE_COLS, FEATURES_PATH, MODEL_PATH, TEST_DATA_PATH, TEST_TARGET_PATH,
    TRAIN_DATA_PATH, TRAIN_TARGET_PATH,
)


def load(path):
    if not os.path.exists(path):
        pytest.skip(f"{os.path.basename(path)} not available")
    with open(path, 'rb') as f:
        try:
            return pickle.load(f)
        except Exception as exc:  # e.g. an un-pulled Git LFS pointer file
            pytest.skip(f"{os.path.basename(path)} could not be loaded: {exc}")


def test_saved_matrices_use_the_feature_list():
    X_test = load(TEST_DATA_PATH)
    assert list(X_test.columns) == FEATURE_COLS
    assert not X_test.isna().any().any()


def test_lookup_artifacts_are_well_formed():
    art = load(FEATURE_ARTIFACTS_PATH)
    table = art['zip_table']
    assert {'lat', 'lon', 'median', 'std', 'count'} <= set(table.columns)
    assert not table.isna().any().any()
    ref = np.asarray(art['flood_proxy_sorted'])
    assert (np.diff(ref) >= 0).all(), "flood reference must be sorted ascending"


def test_end_to_end_prediction_is_sane():
    model, art = load(MODEL_PATH), load(FEATURE_ARTIFACTS_PATH)
    lat, lon = 25.77, -80.19  # downtown Miami
    zip_code = features.nearest_zip(art['zip_table'], lat, lon)
    row = features.build_feature_row(3, 2, 1500, 30, 0, 0, lat, lon,
                                     art['zip_table'].loc[zip_code], art['flood_proxy_sorted'])
    price = float(np.expm1(model.predict(row)[0]))
    assert 100_000 < price < 5_000_000


def test_model_scores_better_than_chance_on_saved_test_set():
    model = load(MODEL_PATH)
    X_test, y_test = load(TEST_DATA_PATH), load(TEST_TARGET_PATH)
    pred = model.predict(X_test)
    r2 = 1 - ((y_test - pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
    assert r2 > 0.6


def test_saved_features_match_a_fresh_rebuild():
    """Re-running step 2 on step 1's output must reproduce the saved training and test files"""
    cleaned = load(FEATURES_PATH)
    out = features.build_datasets(cleaned)
    pd.testing.assert_frame_equal(out['X_train'], load(TRAIN_DATA_PATH))
    pd.testing.assert_frame_equal(out['X_test'], load(TEST_DATA_PATH))
    pd.testing.assert_series_equal(out['y_train'], load(TRAIN_TARGET_PATH))
    pd.testing.assert_series_equal(out['y_test'], load(TEST_TARGET_PATH))
