"""
Reproducibility: the same inputs and seed must give the same outputs, so re-running the
pipeline reproduces the committed model instead of quietly producing a different one.
"""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

import features
from real_estate_config import LIGHTGBM_PARAMS, MODEL_METADATA_PATH


def frame():
    return pd.DataFrame({'lat': np.linspace(25.5, 26.2, 50), 'lon': np.linspace(-80.5, -80.1, 50)})


def test_coordinate_jitter_is_deterministic():
    a = features.add_coordinate_jitter(frame())
    b = features.add_coordinate_jitter(frame())
    pd.testing.assert_frame_equal(a, b)


def test_coordinate_jitter_changes_with_seed_and_stays_small():
    base = frame()
    a = features.add_coordinate_jitter(base, seed=1)
    b = features.add_coordinate_jitter(base, seed=2)
    assert not np.allclose(a['lat'], b['lat'])
    assert (a['lat'] - base['lat']).abs().max() < 6 * features.JITTER_DEGREES
    assert not np.allclose(a['lat'], base['lat'])  # it actually adds noise


def test_jitter_does_not_modify_its_input():
    base = frame()
    before = base.copy()
    features.add_coordinate_jitter(base)
    pd.testing.assert_frame_equal(base, before)


def test_model_training_is_deterministic():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 5)), columns=list('abcde'))
    y = X['a'] * 2 + rng.normal(size=200)
    preds = [lgb.train(LIGHTGBM_PARAMS, lgb.Dataset(X, y), num_boost_round=50).predict(X) for _ in range(2)]
    np.testing.assert_array_equal(preds[0], preds[1])


def test_saved_metadata_has_the_expected_fields():
    if not os.path.exists(MODEL_METADATA_PATH):
        pytest.skip("model_metadata.json not available")
    with open(MODEL_METADATA_PATH, encoding='utf-8') as f:
        meta = json.load(f)
    assert {'python', 'lightgbm', 'pandas', 'numpy', 'scikit_learn', 'random_state', 'metrics'} <= set(meta)
    assert meta['n_features'] == len(meta['features'])
