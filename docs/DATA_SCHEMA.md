# Data Schema & Feature Definitions

## Raw dataset

**Source**: Kaggle, Florida Real Estate Sold 2026
**Records**: 10,893 statewide; 990 after filtering to Miami-Dade + Broward and cleaning

| Column | Type | Null % (raw) | Description |
|--------|------|--------------|-------------|
| `type` | string | 0.0 | Property category (`single_family`, `condos`, `townhomes`, `land`, ...) |
| `sub_type` | string | 77.3 | Detail for some rows (`condo`, `townhouse`); used for the type dummies |
| `listPrice` | float | 0.1 | Listing price (USD), not used as a feature |
| `lastSoldPrice` | int | 0.0 | Sale price (USD), the **target** |
| `sqft` | float | 6.9 | Living area (sq ft) |
| `stories` | float | 15.9 | Number of stories (unused) |
| `beds` | float | 6.2 | Bedrooms |
| `baths` | float | 5.9 | Bathrooms |
| `baths_full`, `baths_full_calc` | float | 6.3 | Full-bath counts (unused, redundant with `baths`) |
| `garage` | float | 32.9 | Garage spaces (unused) |
| `year_built` | float | 6.2 | Construction year, renamed `yr_built` |
| `zip` | float in CSV | 0.0 | ZIP code; normalized to a 5-digit string |
| `sanitized_text` | string | 0.3 | Listing description (unused) |

Columns added in step 1: `county_name`, `lat`, `lon` (from `pgeocode` ZIP centroids), `flood_risk` (constant `'X'`).

---

## Modeling features (16)

The model uses exactly the columns in `FEATURE_COLS` (`src/real_estate_config.py`), in this order. **No feature is computed from the sale price of the row it describes.**

### Property (7)

| Feature | Definition | Notes |
|---------|-----------|-------|
| `beds` | Raw bedroom count | Range 0–9 |
| `baths` | Raw bathroom count | Range 1–9 |
| `sqft` | Living area | Range 440–7,631, median 1,456 |
| `sqft_log` | `log1p(sqft)` | Captures diminishing returns on size |
| `property_age` | `2026 - yr_built` | Range -1–101 (a few pre-construction sales have year 2027) |
| `is_condo` | 1 if `sub_type == 'condo'` | ~38% of rows |
| `is_townhouse` | 1 if `sub_type == 'townhouse'` | ~12% of rows |

### Geospatial (5)

Coordinates are **ZIP-code centroids** from `pgeocode` with ±0.005° Gaussian jitter. They are not property addresses.

| Feature | Definition |
|---------|-----------|
| `lat` | Latitude (25.28–26.32) |
| `lon` | Longitude (-80.63 to -80.09) |
| `dist_downtown` | Geodesic miles to downtown Miami (25.7617, -80.1918) |
| `dist_brickell` | Geodesic miles to Brickell (25.7582, -80.1911) |
| `dist_beach` | Geodesic miles to Miami Beach (25.7945, -80.1298) |

Distances to these three Miami landmarks reach 40+ miles for Broward properties; they are a rough position proxy, not a measure of beach or downtown access there.

### Neighborhood aggregations (3) — training sales only

Computed per ZIP **after** the train/test split:

| Feature | Definition |
|---------|-----------|
| `neighborhood_median_price` | Median sale price of training sales in the ZIP |
| `neighborhood_price_std` | Std of training sale prices in the ZIP |
| `neighborhood_sales_count` | Number of training sales in the ZIP |

How each row gets its values:
- **Test rows** (and dashboard predictions): stats from all training sales in the row's ZIP.
- **Training rows**: leave-one-out, i.e. the same stats with the row's own sale excluded, so the model can't learn from its own target.
- **ZIPs with no (other) training sales**: global training median and std, with count 0. A ZIP with exactly one other sale uses the global std (a single value has no spread).

### Risk (1)

| Feature | Definition |
|---------|-----------|
| `flood_risk_percentile` | Percentile rank of `max(25.8 - lat, 0)`. A crude latitude proxy; many properties north of 25.8° tie at the minimum (0.35). Not real flood-zone data. |

### Removed features

| Feature | Why removed |
|---------|-------------|
| `price_per_sqft` | `lastSoldPrice / sqft` contains the target |
| `list_to_sold_ratio` | `lastSoldPrice / listPrice` contains the target |
| `sale_year`, `sale_month` | Constants (no sale date in the data), so no information |

`near_coast`, `property_age_ordinal`, and `neighborhood_price_tier` were previously computed in step 2 but never used by the model; they are no longer computed.

---

## Target

`lastSoldPrice` (USD) after filtering to $50K–$10M:

| Statistic | Value |
|-----------|-------|
| Mean | $773,517 |
| Median | $529,250 |
| Std | $956,133 |
| Min / Max | $65,000 / $9,900,000 |

The model is trained on `log1p(lastSoldPrice)` (mean 13.19, std 0.80). Convert predictions back with `np.expm1`. Errors in log space approximate percentage errors: a log MAE of 0.244 ≈ 27.6%.

---

## Pickle files

| File | Shape / contents |
|------|------------------|
| `data/features_engineered.pkl` | (990, 18) cleaned frame from step 1 |
| `data/X_train.pkl` / `y_train.pkl` | (792, 16) / (792,): training features / log target |
| `data/X_test.pkl` / `y_test.pkl` | (198, 16) / (198,): test features / log target |
| `data/feature_artifacts.pkl` | Dict: `zip_table` (per-ZIP centroid lat/lon plus training median/std/count for 128 ZIPs), `global_stats` (fallback median/std/count), `flood_proxy_sorted` (sorted latitude-proxy values for percentile lookups) |
| `models/lightgbm_miami_v1.pkl` | Trained LightGBM model |

```python
import pickle, numpy as np
X_test = pickle.load(open('data/X_test.pkl', 'rb'))
y_test = pickle.load(open('data/y_test.pkl', 'rb'))
prices = np.expm1(y_test)
```

Pickles are version-sensitive: load them with the same pandas version that wrote them (re-run `python run_pipeline.py` after upgrading).

---

## Feature statistics (all 990 rows, train + test)

```
Feature                        Mean        Std        Min        Max
beds                           2.85       1.13       0.00       9.00
baths                          2.44       1.04       1.00       9.00
sqft                         1,671        864        440      7,631
sqft_log                       7.31       0.45       6.09       8.94
property_age                   42.4       21.2       -1.0      101.0
is_condo                       0.38       0.49       0.00       1.00
is_townhouse                   0.12       0.32       0.00       1.00
lat                           25.94      0.227      25.28      26.32
lon                          -80.25      0.106     -80.63     -80.09
dist_downtown                  18.7       10.4       0.11       43.0
dist_brickell                  18.9       10.4       0.00       42.9
dist_beach                     18.9       9.64       0.25       47.3
neighborhood_median_price   643,269    475,684    120,000  4,625,000
neighborhood_price_std      538,052    615,865          0  4,085,410
neighborhood_sales_count       7.9        4.8        0.0       24.0
flood_risk_percentile          0.50       0.24       0.35       1.00
```

---

## Validation rules (step 1)

| Rule | Action |
|------|--------|
| County in `TARGET_COUNTIES` (Miami-Dade, Broward) | Drop otherwise |
| Coordinates within `METRO_LAT_BOUNDS` / `METRO_LON_BOUNDS` | Drop otherwise (bad geocodes) |
| Price $50K–$10M, sqft 400–10,000 | Drop outside |
| Missing beds, baths, sqft, yr_built, price, lat, or lon | Drop row |
| Required columns present | Raise an error if not |

---

## Possible data improvements

1. FEMA flood zones to replace the latitude proxy
2. Real sale dates for seasonality and appreciation
3. Address-level geocoding instead of ZIP centroids
4. Property condition, lot size, HOA fees, taxes, amenities
5. More sales per ZIP (or shrinkage) for steadier neighborhood stats
