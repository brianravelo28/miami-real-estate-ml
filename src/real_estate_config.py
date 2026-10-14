"""
Real Estate Project Configuration
Single source of truth for all paths, parameters, constants
"""

import os

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Data files
RAW_DATA_PATH = os.path.join(DATA_DIR, 'florida_sold_2026.csv')
FEATURES_PATH = os.path.join(DATA_DIR, 'features_engineered.pkl')
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'X_train.pkl')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'X_test.pkl')
TRAIN_TARGET_PATH = os.path.join(DATA_DIR, 'y_train.pkl')
TEST_TARGET_PATH = os.path.join(DATA_DIR, 'y_test.pkl')
# Train-only ZIP stats and other lookups the dashboard needs at prediction time
FEATURE_ARTIFACTS_PATH = os.path.join(DATA_DIR, 'feature_artifacts.pkl')

# Model files
MODEL_PATH = os.path.join(MODELS_DIR, 'lightgbm_miami_v1.pkl')
# Library versions and settings the model was built with (the dashboard compares against these)
MODEL_METADATA_PATH = os.path.join(MODELS_DIR, 'model_metadata.json')

# Report files
SHAP_SUMMARY_PATH = os.path.join(REPORTS_DIR, 'shap_summary.png')
RESIDUAL_PLOT_PATH = os.path.join(REPORTS_DIR, 'residuals.png')

# ============================================================================
# GEOSPATIAL REFERENCE POINTS (Miami)
# ============================================================================
DOWNTOWN_MIAMI = (25.7617, -80.1918)
BRICKELL = (25.7582, -80.1911)
MIAMI_BEACH = (25.7945, -80.1298)

# Target counties for filtering statewide data (matched via real ZIP-code geocoding,
# not ZIP prefixes -- prefixes like 334 bleed into Palm Beach/Monroe counties)
TARGET_COUNTIES = ['Miami-Dade', 'Broward']

# Sanity bounding box around Miami-Dade + Broward, to catch occasional bad
# ZIP-to-county geocodes (e.g. a ZIP mislabeled with the right county name
# but coordinates on Florida's Gulf coast)
METRO_LAT_BOUNDS = (25.10, 26.50)
METRO_LON_BOUNDS = (-80.95, -79.90)

# ============================================================================
# DATA PROCESSING
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_PRICE = 50_000  # Remove sales < $50K
MAX_PRICE = 10_000_000  # Remove sales > $10M
MIN_SQFT = 400  # Remove properties < 400 sqft
MAX_SQFT = 10_000  # Remove properties > 10K sqft

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
REFERENCE_YEAR = 2026

# Pulls a ZIP's price stats toward its county's: weight n/(n+k) on the ZIP's own n sales.
# 0 = off (raw ZIP stats). 20 = the largest value whose paired cross-validated accuracy cost
# stayed within one standard error (see the shrinkage study in src/05_evaluate_models.py).
NEIGHBORHOOD_SHRINKAGE_K = 20

# Dashboard: a pin farther than this from the nearest ZIP centroid with sales is treated as
# outside the area we have data for. Urban ZIP centroids are ~2 miles apart (p90 ~3 miles).
FAR_FROM_DATA_MILES = 5.0

# ============================================================================
# MODEL PARAMETERS (LightGBM)
# ============================================================================
LIGHTGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
}

LIGHTGBM_NUM_ROUNDS = 200

# ============================================================================
# FEATURE LIST (For modeling)
# ============================================================================
FEATURE_COLS = [
    # Property features
    'beds', 'baths', 'sqft', 'sqft_log', 'property_age', 'is_condo', 'is_townhouse',

    # Geospatial
    'lat', 'lon', 'dist_downtown', 'dist_brickell', 'dist_beach',

    # Neighborhood (computed from training sales only; see 02_engineer_features.py)
    'neighborhood_median_price', 'neighborhood_price_std', 'neighborhood_sales_count',

    # Flood risk (latitude proxy)
    'flood_risk_percentile',
]

# ============================================================================
# SUCCESS CRITERIA
# ============================================================================
TARGET_R2 = 0.80
TARGET_RMSE = 0.20
TARGET_MAE = 0.18
