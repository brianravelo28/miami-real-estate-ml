# Methodology & ML Approach

## Objective

Predict residential sale prices in Miami-Dade and Broward counties (2026) with:
1. **Predictive accuracy** (targets: R² ≥ 0.80, log-scale MAE ≤ 0.18)
2. **Explainability** (SHAP values for every prediction)
3. **Geospatial reasoning** (location-aware features)

---

## Data Overview

### Source
- **Dataset**: Florida Real Estate Sold 2026 (Kaggle), 10,893 closed sales statewide
- **Target geography**: Miami-Dade and Broward counties, selected by geocoding each ZIP with `pgeocode` and filtering on county name
- **Final sample**: 990 properties (503 Miami-Dade, 487 Broward) across 128 ZIP codes
- **Price range**: $50K–$10M after outlier removal (actual: $65K–$9.9M)
- **Sqft range**: 400–10,000 after outlier removal (actual: 440–7,631)

### Target variable
- **Raw**: `lastSoldPrice` (USD)
- **Modeled**: `log1p(lastSoldPrice)`, because prices are strongly right-skewed

Raw prices (after filtering):
```
Count:    990
Mean:     $773,517
Median:   $529,250
Std:      $956,133
Min/Max:  $65,000 / $9,900,000
Skewness: 4.6
```

Log scale:
```
Mean: 13.19   Std: 0.80   Skewness: 0.41
```

---

## Data quality issues addressed

| Issue | Root cause | Solution |
|-------|-----------|----------|
| No coordinates | Dataset only has ZIP codes | Geocode each ZIP to its centroid with `pgeocode`, plus ±0.005° jitter |
| ZIP prefixes bleed across counties | e.g. `334xx` includes Palm Beach and Monroe | Filter on geocoded county name instead of prefix |
| Bad ZIP geocodes | e.g. ZIP 33973 labeled Broward but located in Lee County | Bounding-box sanity filter |
| ZIPs stored as floats | `33446.0` | Normalize to 5-digit strings before geocoding |
| Property type as text | No numeric encoding | `is_condo`, `is_townhouse` dummies |
| No flood data | No FEMA shapefile | Latitude-based proxy |
| Extreme prices / sqft | Entry errors, non-residential | Keep $50K–$10M and 400–10,000 sqft |
| No sale date | Not in dataset | No temporal features are used |

---

## Feature engineering

Domain-driven features rather than automated generation. Full definitions are in [DATA_SCHEMA.md](DATA_SCHEMA.md).

| Group | Features |
|-------|----------|
| Property | beds, baths, sqft, sqft_log, property_age, is_condo, is_townhouse |
| Geospatial | lat, lon, dist_downtown, dist_brickell, dist_beach |
| Neighborhood (by ZIP) | neighborhood_median_price, neighborhood_price_std, neighborhood_sales_count (training sales only) |
| Risk | flood_risk_percentile (latitude proxy) |

### Avoiding target leakage

An earlier version of this project included `price_per_sqft` (sale price ÷ sqft) and `list_to_sold_ratio` (sale price ÷ list price), computed ZIP statistics over all sales before splitting, and reported R² = 0.992 with 4.6% error. Those numbers were inflated because the features contained the answer. The current pipeline:

1. **Drops** `price_per_sqft` and `list_to_sold_ratio` (both derive from the sale price) and the constant `sale_year`/`sale_month`.
2. **Splits first** (80/20, stratified by price quartile), then computes anything that touches sale prices.
3. Computes ZIP price stats from **training rows only**. Test rows see stats from all training sales in their ZIP; training rows see **leave-one-out** stats (their own sale excluded) so the model can't memorize its own target.
4. Falls back to global training stats for ZIPs with no other training sales.

The result is a much lower but meaningful score (below). The dashboard uses the same lookups at prediction time, so what it does matches what was evaluated.

---

## Model selection

LightGBM was chosen because it trains in seconds on ~1K rows, handles numeric features without preprocessing, and has exact tree-based SHAP support.

| Model | Verdict |
|-------|---------|
| Linear regression | Can't capture non-linearities; baseline only |
| Random forest / XGBoost | Comparable, heavier; not benchmarked in this project |
| **LightGBM** | **Selected** |
| Neural network | Overkill for ~1K samples |

### Hyperparameters

```python
LIGHTGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
}
LIGHTGBM_NUM_ROUNDS = 200
```

These are reasonable defaults, not the result of a systematic search, and there is no validation set or early stopping. With only 198 test rows, cross-validation would give a steadier estimate and is listed under future work.

---

## Model evaluation

### Test-set performance (198 held-out properties)

```
RMSE (log scale): 0.3352   (≈ 40% in price terms)
MAE  (log scale): 0.2440   (≈ 27.6% average price error)
R² (test):        0.8261
R² (train):       0.9681
```

| Target | Goal | Result |
|--------|------|--------|
| R² | ≥ 0.80 | 0.826 (met) |
| RMSE (log) | ≤ 0.20 | 0.335 (not met) |
| MAE (log) | ≤ 0.18 | 0.244 (not met) |

The train/test gap (0.968 vs 0.826) indicates overfitting on 792 training rows. The 10th–90th percentile of test errors spans roughly ×0.70 to ×1.63 of the predicted price, which is what the dashboard reports as its "typical range".

### Feature importance

Top features by LightGBM gain:

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `sqft` | 2027 |
| 2 | `sqft_log` | 336 |
| 3 | `neighborhood_median_price` | 266 |
| 4 | `baths` | 225 |
| 5 | `dist_beach` | 196 |
| 6 | `neighborhood_price_std` | 194 |
| 7 | `lon` | 192 |
| 8 | `dist_downtown` | 184 |
| 9 | `is_condo` | 155 |
| 10 | `lat` | 150 |

By mean |SHAP| (test set) the top features are `sqft` (0.35), `is_condo` (0.10), `lon` (0.09), `dist_downtown` (0.07), and `sqft_log` (0.06). Size dominates, followed by property type and location.

---

## Explainability

`shap.TreeExplainer` computes exact SHAP values for the LightGBM model. The dashboard shows a per-prediction bar chart of the top 10 contributions (Tab 1) and global mean |SHAP| importance (Tab 3). See `reports/SHAP_INTERPRETATION_GUIDE.md` for how to read them.

---

## Limitations & biases

### Accuracy
- Average error is ~28% and the 80% error band is roughly -30% to +63%. Treat predictions as rough estimates.
- Train R² (0.97) is far above test R² (0.83): the model overfits the small training set.
- The test estimate itself is noisy (198 rows, one random split).
- ZIP-level stats come from ~6 training sales per ZIP on average; four ZIPs have none and use the global fallback.
- Leave-one-out ZIP stats are a mild approximation: training rows see n-1 neighbors while test rows see n.

### Data limitations
- **Geographic**: Miami-Dade and Broward only
- **Temporal**: single-year snapshot, no sale dates
- **Location precision**: ZIP centroids with small jitter, not property addresses
- **Missing features**: no HOA fees, taxes, condition, views, or lot size
- **Small neighborhoods**: ~6 training sales per ZIP on average (4 of 128 ZIPs have none), so ZIP aggregates are noisy

### Model limitations
- ~1K samples and 16 features, so overfitting is a risk and the test estimate is noisy
- Poor extrapolation outside the training range (very large or very small homes)
- No prediction intervals; the dashboard's ±6% range is a fixed heuristic, not a statistical interval

### Potential biases
- ZIP-level aggregation can reinforce existing price disparities
- Only completed sales are included (no failed listings)
- The Kaggle sample may not be representative of all sales in these counties

---

## Future enhancements

**Near term**
1. k-fold cross-validation and a small hyperparameter search (narrow the train/test gap)
2. Smoothed/shrunk ZIP statistics for sparsely sampled ZIPs
3. Real flood-zone data (FEMA) in place of the latitude proxy
4. Prediction intervals via quantile regression

**Longer term**
1. Historical sales with real dates for seasonality and trend features
2. Address-level geocoding
3. Additional markets
4. Model monitoring and scheduled retraining

---

## References

- [SHAP Documentation](https://shap.readthedocs.io/)
- [LightGBM Parameters](https://lightgbm.readthedocs.io/)
- [Kaggle dataset](https://www.kaggle.com/datasets/kanchana1990/florida-real-estate-sold-dataset-2026)
- [pgeocode](https://github.com/symerio/pgeocode)
