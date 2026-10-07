# System Architecture

End-to-end architecture of the Miami Real Estate ML pipeline, from raw data to the interactive dashboard.

---

## Data Flow

```
Kaggle CSV: Florida Real Estate Sold 2026 (10,893 statewide records)
        │
  STEP 1: LOAD & VALIDATE  (src/01_load_data.py)
   - Clean ZIPs to 5-digit strings
   - Geocode every ZIP with pgeocode → lat, lon, county
   - Keep Miami-Dade + Broward only (+ sanity bounding box)
   - Remove price/sqft outliers, drop rows missing critical fields
   - Add small jitter to coordinates
        │
  data/features_engineered.pkl  (990 records)
        │
  STEP 2: FEATURE ENGINEERING  (src/02_engineer_features.py)
   - Property, geospatial, and flood-proxy features (none use the sale price)
   - 80/20 split → 792 train / 198 test (BEFORE any price-based statistic)
   - ZIP price stats computed from training rows only
   - 16-feature matrix, log1p(price) target
        │
  data/X_train.pkl, y_train.pkl, X_test.pkl, y_test.pkl, feature_artifacts.pkl
        │
  STEP 3: TRAIN LIGHTGBM  (src/03_train_model.py)
   - 200 boosting rounds
   - Metrics, residual and actual-vs-predicted plots
        │
  models/lightgbm_miami_v1.pkl, reports/evaluation_report.txt
        │
  STEP 4: SHAP ANALYSIS  (src/04_shap_analysis.py)
   - Summary, importance, force, and dependence plots
        │
  STREAMLIT DASHBOARD  (app.py)
   - Tab 1: Search & Predict
   - Tab 2: Neighborhood Map
   - Tab 3: Diagnostics
```

`src/05_build_dashboard.py` is the original Dash version of the dashboard and is superseded by `app.py`.

---

## Components

### 1. Data ingestion (`src/01_load_data.py`)

**Input**: `data/florida_sold_2026.csv` (statewide), config from `src/real_estate_config.py`.

1. Load the CSV.
2. Normalize `zip` to a 5-digit string (the raw column is float, e.g. `33446.0`).
3. Geocode each unique ZIP with `pgeocode` to get `latitude`, `longitude`, and `county_name`.
4. Keep rows whose county is in `TARGET_COUNTIES` (`Miami-Dade`, `Broward`). Filtering on county rather than ZIP prefix matters: prefixes like `334` also cover Palm Beach and Monroe counties.
5. Drop rows whose coordinates fall outside the sanity box `METRO_LAT_BOUNDS` / `METRO_LON_BOUNDS` (catches bad ZIP geocodes, e.g. ZIP 33973 is labeled Broward but sits in Lee County).
6. Remove outliers: price $50K–$10M, sqft 400–10,000.
7. Drop rows missing beds, baths, sqft, year built, price, or coordinates.
8. Add ±0.005° (≈0.35 mi) Gaussian jitter to coordinates so same-ZIP sales don't stack.
9. Add `flood_risk = 'X'` (no FEMA data available).

**Output**: `data/features_engineered.pkl`, 990 records (503 Miami-Dade, 487 Broward) across 128 ZIP codes.

### 2. Feature engineering (`src/02_engineer_features.py`)

Builds the 16 modeling features listed in `FEATURE_COLS` (see [DATA_SCHEMA.md](DATA_SCHEMA.md)):

- **Property**: beds, baths, sqft, sqft_log, property_age, is_condo, is_townhouse
- **Geospatial**: lat, lon, distances to downtown Miami, Brickell, and Miami Beach (geodesic miles)
- **Neighborhood** (grouped by ZIP): median price, price std, sales count, from **training sales only**
- **Risk**: flood_risk_percentile (latitude-based proxy)

Leakage safeguards:
- No feature is computed from the sale price of the row it describes (an earlier version used `price_per_sqft` and `list_to_sold_ratio`, which leaked the target).
- The split (80/20, `random_state=42`, stratified by price quartile) happens **before** any price-based statistic is computed.
- Training rows get **leave-one-out** ZIP stats (their own sale is excluded); test rows and dashboard predictions use stats from all training sales in the ZIP. ZIPs with no other training sales fall back to the global training median/std with a count of 0.
- Constant columns (`sale_year`, `sale_month`) are no longer used.

The target is `log1p(lastSoldPrice)`.

`data/feature_artifacts.pkl` stores what the dashboard needs at prediction time: the per-ZIP training stats and ZIP centroids, the global fallback stats, and the sorted latitude-proxy values used to compute `flood_risk_percentile`.

### 3. Model training (`src/03_train_model.py`)

LightGBM regression, hyperparameters from `LIGHTGBM_PARAMS` in the config (`num_leaves=31`, `learning_rate=0.05`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=5`), 200 rounds.

Test-set results:

| Metric | Value |
|--------|-------|
| RMSE (log scale) | 0.3352 |
| MAE (log scale) | 0.2440 (≈ 27.6% price error) |
| R² (test) | 0.8261 |
| R² (train) | 0.9681 |

The gap between train and test R² shows some overfitting on the small training set. The 10th–90th percentile of test errors corresponds to roughly ×0.70 to ×1.63 of the predicted price, which the dashboard shows as its "typical range".

Top features by gain: `sqft` (2027), `sqft_log` (336), `neighborhood_median_price` (266), `baths` (225), `dist_beach` (196).

**Outputs**: `models/lightgbm_miami_v1.pkl`, `reports/evaluation_report.txt`, `reports/residuals.png`, `reports/actual_vs_predicted.png`.

### 4. Explainability (`src/04_shap_analysis.py`)

`shap.TreeExplainer` on the test set (198 × 16 SHAP matrix, base value ≈ 13.2 in log space, ≈ $540K). Generates summary, importance, force, and dependence plots; see `reports/SHAP_INTERPRETATION_GUIDE.md`.

### 5. Dashboard (`app.py`)

Streamlit app. On startup it loads the model and test split and computes SHAP values once (`st.cache_resource`).

**Tab 1: Search & Predict**
- Inputs: beds, baths, sqft, property age, property type.
- Location: a crosshair fixed at the center of a Folium map; panning the map sets the point. The crosshair is red when the map is at zoom ≥ 10 and centered within the Miami-Dade/Broward bounding box, gray otherwise.
- Output: predicted price, a "typical range" (10th–90th percentile of the model's test errors), and a SHAP bar chart of the top 10 features for that prediction.
- Neighborhood features are looked up from the nearest ZIP centroid in `feature_artifacts.pkl` (the same training-only stats the model saw); distances and the flood percentile are computed from the pin the same way as in training.

**Tab 2: Neighborhood Map**
- Folium map of 100 sampled test properties, colored by predicted price.
- Median, mean, and range stats recompute for the properties inside the current map view.

**Tab 3: Diagnostics**
- R², MAE, RMSE; residual plot; global SHAP feature importance (mean |SHAP|, top 10).

All Plotly charts carry a drag-to-zoom / double-click-to-reset hint.

---

## Deployment

Deployed on Render's free tier (auto-deploys on push to `main`): https://miami-real-estate-ml.onrender.com/. The app needs only `app.py`, the model pickle, and `data/X_test.pkl` / `y_test.pkl`. See [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Pipeline robustness

| Step | Failure mode | Mitigation |
|------|--------------|-----------|
| Load | Missing CSV | Clear error with Kaggle download instructions |
| Load | Float-typed ZIPs / bad geocodes | ZIP normalization; county filter plus bounding-box sanity check |
| Load | Missing required columns | Validated up front |
| Engineer | NaN in features | Filled with median/default |
| Dashboard | Relative-path breakage | Paths resolved from `app.py`'s own location |
| Dashboard | Map returns null bounds | Validity check before using bounds |

---

## Future enhancements

1. Cross-validation and a small hyperparameter search to narrow the train/test gap
2. Proper prediction intervals via quantile regression (the current range is empirical)
3. FEMA flood zones instead of the latitude proxy
4. Real sale dates for seasonality and trend features
5. Monitoring for prediction drift once real usage exists
