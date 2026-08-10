"""
Real Estate Project Configuration
Single source of truth for all paths, parameters, constants
"""

import os

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
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

# Model files
MODEL_PATH = os.path.join(MODELS_DIR, 'lightgbm_miami_v1.pkl')

# Report files
SHAP_SUMMARY_PATH = os.path.join(REPORTS_DIR, 'shap_summary.png')
SHAP_FORCE_EXAMPLE_PATH = os.path.join(REPORTS_DIR, 'shap_force_example.png')
RESIDUAL_PLOT_PATH = os.path.join(REPORTS_DIR, 'residuals.png')

# ============================================================================
# GEOSPATIAL REFERENCE POINTS (Miami)
# ============================================================================
DOWNTOWN_MIAMI = (25.7617, -80.1918)
BRICKELL = (25.7582, -80.1911)
MIAMI_BEACH = (25.7945, -80.1298)

# Miami metro ZIP code prefixes (for filtering statewide data)
MIAMI_ZIP_PREFIXES = [
    '331',  # Miami, Coral Gables, Wynwood
    '334',  # Miami Beach, Sunny Isles
]

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

# Binning edges for property age
PROPERTY_AGE_BINS = [0, 5, 20, 40, 200]
PROPERTY_AGE_LABELS = ['new', 'recent', 'established', 'old']

# Flood zone encoding (ordinal: lower = less risk)
FLOOD_ZONE_ENCODING = {
    'X': 0,    # No risk
    'A': 1,    # Risk zones
    'AE': 2,
    'VE': 3,   # Highest risk (coastal high hazard)
}

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
    'beds', 'baths', 'sqft', 'sqft_log', 'property_age', 'price_per_sqft',
    'list_to_sold_ratio', 'is_condo', 'is_townhouse',

    # Geospatial
    'lat', 'lon', 'dist_downtown', 'dist_brickell', 'dist_beach',

    # Neighborhood
    'neighborhood_median_price', 'neighborhood_price_std', 'neighborhood_sales_count',

    # Flood risk
    'flood_risk_percentile',

    # Temporal
    'sale_year', 'sale_month',
]

# ============================================================================
# DASHBOARD CONFIG
# ============================================================================
DASH_HOST = '0.0.0.0'
DASH_PORT = 7860
DASH_DEBUG = True

# ============================================================================
# SUCCESS CRITERIA
# ============================================================================
TARGET_R2 = 0.80
TARGET_RMSE = 0.20
TARGET_MAE = 0.18
