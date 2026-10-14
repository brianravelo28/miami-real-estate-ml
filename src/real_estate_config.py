"""
Real Estate Project Configuration
Single source of truth for paths and cleaning rules
"""

import os

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

os.makedirs(DATA_DIR, exist_ok=True)

RAW_DATA_PATH = os.path.join(DATA_DIR, 'florida_sold_2026.csv')
# Cleaned Miami-Dade + Broward sales: the only data file the dashboard reads
CLEAN_DATA_PATH = os.path.join(DATA_DIR, 'sales_clean.csv')

# Columns kept in the cleaned file (the listing description text is not needed)
OUTPUT_COLS = [
    'type', 'sub_type', 'listPrice', 'lastSoldPrice', 'sqft', 'beds', 'baths',
    'yr_built', 'zip', 'county_name', 'lat', 'lon',
]

# ============================================================================
# GEOGRAPHY
# ============================================================================
# Target counties for filtering statewide data (matched via real ZIP-code geocoding,
# not ZIP prefixes -- prefixes like 334 bleed into Palm Beach/Monroe counties)
TARGET_COUNTIES = ['Miami-Dade', 'Broward']

# Sanity bounding box around Miami-Dade + Broward, to catch occasional bad
# ZIP-to-county geocodes (e.g. a ZIP mislabeled with the right county name
# but coordinates on Florida's Gulf coast)
METRO_LAT_BOUNDS = (25.10, 26.50)
METRO_LON_BOUNDS = (-80.95, -79.90)

# ============================================================================
# CLEANING RULES
# ============================================================================
MIN_PRICE = 50_000  # Remove sales < $50K
MAX_PRICE = 10_000_000  # Remove sales > $10M
MIN_SQFT = 400  # Remove properties < 400 sqft
MAX_SQFT = 10_000  # Remove properties > 10K sqft

# ============================================================================
# DASHBOARD
# ============================================================================
# A ZIP with fewer sales than this is shown but flagged: its median is not a neighborhood price
MIN_ZIP_SALES = 5
