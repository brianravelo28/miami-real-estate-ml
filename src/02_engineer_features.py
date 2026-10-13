"""
Step 2: Feature Engineering
Builds the modeling features and train/test split, and saves them with the lookups
the dashboard needs. The logic lives in features.py so the dashboard uses the same code.

Leakage rules (enforced in features.py and covered by tests/test_features.py):
  - No feature is derived from the sale price of the row it describes.
  - The split happens BEFORE any price-based neighborhood statistic.
  - Training rows get leave-one-out ZIP stats; test rows use training-only ZIP stats.
"""

import logging
import pickle

from real_estate_config import *
import features

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_cleaned_data():
    """Load output from step 1"""
    logger.info("Loading cleaned data...")
    df = pickle.load(open(FEATURES_PATH, 'rb'))
    logger.info(f"Loaded {len(df):,} records")
    return df


def save_artifacts(out):
    """Save train/test data, target, and dashboard lookups"""
    pickle.dump(out['X_train'], open(TRAIN_DATA_PATH, 'wb'))
    pickle.dump(out['X_test'], open(TEST_DATA_PATH, 'wb'))
    pickle.dump(out['y_train'], open(TRAIN_TARGET_PATH, 'wb'))
    pickle.dump(out['y_test'], open(TEST_TARGET_PATH, 'wb'))
    pickle.dump({
        'zip_table': out['zip_table'],
        'global_stats': out['global_stats'],
        'flood_proxy_sorted': out['flood_proxy_sorted'],
    }, open(FEATURE_ARTIFACTS_PATH, 'wb'))

    logger.info(f"\n✓ Saved X_train to {TRAIN_DATA_PATH}")
    logger.info(f"✓ Saved X_test to {TEST_DATA_PATH}")
    logger.info(f"✓ Saved y_train to {TRAIN_TARGET_PATH}")
    logger.info(f"✓ Saved y_test to {TEST_TARGET_PATH}")
    logger.info(f"✓ Saved ZIP lookup artifacts to {FEATURE_ARTIFACTS_PATH}")


def main():
    logger.info("=" * 60)
    logger.info("STEP 2: ENGINEER FEATURES")
    logger.info("=" * 60)

    df = load_cleaned_data()
    out = features.build_datasets(df)

    logger.info(f"\n✓ Feature matrix: {out['X_train'].shape[1]} features")
    logger.info(f"  Features: {', '.join(FEATURE_COLS)}")
    save_artifacts(out)

    logger.info("\n" + "=" * 60)
    logger.info("✓ Step 2 Complete. Ready for model training.")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
