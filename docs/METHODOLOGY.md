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
| No coordinates | Dataset only has ZIP codes | Geocode each ZIP to its centroid with `pgeocode`, plus seeded ±0.005° jitter |
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
4. Falls back to the county's training stats (global if the county is unknown) for ZIPs with no other training sales.

### Neighborhood shrinkage

A ZIP with 2 training sales has a very noisy median. Each ZIP's median and std are therefore blended toward the county's training median/std with weight n/(n+k) on the ZIP's own n sales (`NEIGHBORHOOD_SHRINKAGE_K`; k = 0 turns it off). It is a smoothing heuristic, not an exact Bayesian estimate, and the sales count stays the raw n. Training rows are leave-one-out at both the ZIP and the county level, so no row sees its own price.

How k was chosen (`src/05_evaluate_models.py`, results in `reports/model_comparison.md`): for k from 0 to 40, paired CV R² against k = 0 over the same folds, plus a "swing" measure, how much a holdout house's predicted log price moves when its neighborhood inputs come from the neighboring ZIP instead (the effect of a map pin snapping to the next ZIP). Rule fixed in advance: the largest k whose paired CV change is within one standard error. That picks k = 20.

Result: no measurable accuracy change (paired ΔR² −0.001 ± 0.003) and a modest reduction in swing (median about 10% → 9%, 90th percentile about 34% → 27% in price terms). It is an improvement on average, not a fix: for the dashboard's default map position the gap between the two nearest ZIPs went from 12% to 18% after shrinkage, because the sales-count feature and the remaining ZIP-level differences still move the tree splits.

The result is a much lower but meaningful score (below). The dashboard uses the same lookups at prediction time, so what it does matches what was evaluated.

---

## Model selection

LightGBM was chosen because it trains in seconds on ~1K rows, handles numeric features without preprocessing, and has exact tree-based SHAP support.

| Model | Verdict |
|-------|---------|
| Predict the mean | Baseline: CV R² ≈ 0 |
| Linear (property features only) | Baseline: CV R² 0.61 |
| Ridge (all 16 features) | CV R² 0.76; location/neighborhood help, but linear misses non-linearities |
| Random forest | CV R² 0.82; close to LightGBM |
| **LightGBM** | **CV R² 0.83; selected** (best of those benchmarked, though its edge over random forest is within noise) |
| XGBoost | Not benchmarked (expected to be similar to LightGBM) |
| Neural network | Not tried; overkill for ~1K samples |

Numbers come from `src/05_evaluate_models.py` (5-fold × 3-repeat CV on the 792 development rows; see `reports/model_comparison.md`).

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

These started as reasonable defaults. `src/05_evaluate_models.py` then tried a 40-configuration random search (`num_leaves` 4–31, `min_data_in_leaf` 5–50, `learning_rate` 0.02–0.08, feature/bagging fractions, `lambda_l2` 0–20, 100–600 rounds), scored with repeated 5-fold CV on the development rows only. The best configuration reached CV R² 0.836 vs 0.827 for the current one, a difference smaller than the fold-to-fold standard deviation (≈ 0.03), and it scored 0.827 vs 0.821 on the untouched holdout (differences this small flip sign from run to run). So the current config was kept. There is still no early stopping.

---

## Model evaluation

### Test-set performance (198 held-out properties)

```
RMSE (log scale): 0.3400   (≈ 40% in price terms)
MAE  (log scale): 0.2454   (≈ 27.8% average price error)
R² (test):        0.8210
R² (train):       0.9680
```

| Target | Goal | Result |
|--------|------|--------|
| R² | ≥ 0.80 | 0.821 (met) |
| RMSE (log) | ≤ 0.20 | 0.340 (not met) |
| MAE (log) | ≤ 0.18 | 0.245 (not met) |

The train/test gap (0.968 vs 0.821) indicates overfitting on 792 training rows. The 10th–90th percentile of test errors spans roughly ×0.71 to ×1.63 of the predicted price, which is what the dashboard reports as its "typical range".

### Feature importance

Top features by LightGBM gain:

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `sqft` | 2019 |
| 2 | `sqft_log` | 330 |
| 3 | `neighborhood_median_price` | 330 |
| 4 | `baths` | 248 |
| 5 | `lon` | 201 |
| 6 | `dist_beach` | 183 |
| 7 | `is_condo` | 156 |
| 8 | `dist_downtown` | 146 |
| 9 | `property_age` | 141 |
| 10 | `lat` | 139 |

By mean |SHAP| (test set) the top features are `sqft` (0.35), `is_condo` (0.10), `lon` (0.09), `neighborhood_median_price` (0.08), and `sqft_log` / `dist_downtown` (0.06). Size dominates, followed by property type and location.

---

## Explainability

`shap.TreeExplainer` computes exact SHAP values for the LightGBM model. The dashboard shows a per-prediction bar chart of the top 10 contributions (Tab 1) and global mean |SHAP| importance (Tab 3). See `reports/SHAP_INTERPRETATION_GUIDE.md` for how to read them.

---

## Limitations & biases

### Accuracy
- Average error is ~28% and the 80% error band is roughly -29% to +63%. Treat predictions as rough estimates.
- Train R² (0.97) is far above test R² (0.83): the model overfits the small training set.
- The test estimate itself is noisy (198 rows, one random split).
- ZIP-level stats come from ~6 training sales per ZIP on average; four ZIPs have none and use their county's stats. Shrinkage toward the county (below) dampens the noise but does not remove it.
- Leave-one-out ZIP stats are a mild approximation: training rows see n-1 neighbors while test rows see n.

### Data limitations
- **Geographic**: Miami-Dade and Broward only
- **Temporal**: single-year snapshot, no sale dates
- **Location precision**: ZIP centroids with small seeded jitter, not property addresses. The particular jitter draw moves test R² by a few hundredths (0.826 vs 0.819 in two runs), about the size of the cross-validation noise.
- **Missing features**: no HOA fees, taxes, condition, views, or lot size
- **Small neighborhoods**: ~6 training sales per ZIP on average (4 of 128 ZIPs have none), so ZIP aggregates are noisy
- **Pin-to-ZIP matching is unverifiable**: the dashboard assigns a map pin to the nearest ZIP centroid, but the data has no real addresses (coordinates are ZIP centroids plus jitter), so the nearest-centroid lookup recovers the true ZIP for 192 of 198 test homes for circular reasons. The headline metrics assume the correct ZIP; real pins near a boundary may do worse, and averaging the three nearest ZIPs did not improve accuracy here (R² 0.816 vs 0.820), so it was not adopted.

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
1. More data or better features, since hyperparameter tuning didn't narrow the train/test gap (cross-validation and a small search are done; see `reports/model_comparison.md`)
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
