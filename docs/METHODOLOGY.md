# Methodology

How the data was prepared, what the dashboard computes, and what happened to the earlier predictive model.

## Data preparation (`src/data_prep.py`, `src/01_load_data.py`)

| Issue | Cause | Handling |
|-------|-------|----------|
| No coordinates | The file only has ZIP codes | Look each ZIP up in `pgeocode` (offline database) and use its center point |
| ZIP prefixes bleed across counties | e.g. `334xx` includes Palm Beach and Monroe | Filter on the geocoded **county name**, not the prefix |
| Bad geocode | ZIP 33973 is labeled Broward but located on the Gulf coast | Bounding-box check around the two counties |
| ZIPs stored as floats | `33446.0` | Normalize to zero-padded 5-character strings before geocoding |
| Extreme prices / sizes | Entry errors, non-residential | Keep $50K–$10M and 400–10,000 sqft |
| Missing key fields | Incomplete listings | Drop rows missing beds, baths, sqft, build year, price or location |
| No sale date | Not in the dataset | No time analysis is attempted |

Each rule has a test (`tests/test_data_prep.py`) because each one broke something earlier: the original "Miami" filter used ZIP prefixes and 59% of what it kept was in other counties.

Result: 990 sales, 128 ZIPs, written to `data/sales_clean.csv`.

## What the dashboard computes (`src/summaries.py`)

- **Medians and counts only.** Median sale price, median price per square foot (sale price ÷ sqft), median size, and the number of sales behind each.
- **Property type** groups the raw `type` field: single family, condo, townhome (including the "condo/townhome/rowhome/coop" label), other (multi-family, mobile).
- **Small samples are surfaced, not hidden:** ZIPs with fewer than 5 sales are flagged (`MIN_ZIP_SALES`), grouped bars with fewer than 10 sales are not drawn, and counts are shown next to every figure.
- **No smoothing or modelling.** Medians are raw medians of the sales in each group.

## Caveats when reading it

- Patterns are not causes. Size, bedrooms, type and location move together (condos are smaller and cluster in certain areas), so a gap between two groups is not the effect of that one factor.
- ZIP medians come from a median of 7 sales per ZIP and are noisy.
- Locations are ZIP centers, so the map cannot show where within a ZIP a sale happened.
- One snapshot of 2026 sales with no dates; no trend or seasonality claims.

## History: the removed price predictor

An earlier version trained LightGBM on 16 features and explained it with SHAP. What happened, in order:

1. **Leakage.** The first version reported R² 0.992 (4.6% error). Two features (`price_per_sqft`, `list_to_sold_ratio`) were derived from the sale price, and ZIP price statistics were computed over all sales before the train/test split. The fix dropped the price-derived features, split first, and used training-only (leave-one-out for training rows) ZIP statistics. The honest score was test R² 0.82 (MAE 0.245 in log terms, about 28% typical error, 80% band roughly −29% to +63%), with cross-validated R² 0.827 ± 0.024 and train R² 0.968.
2. **Model comparison.** On 5-fold × 3-repeat cross-validation: predict-the-mean ≈ 0, linear on property features 0.61, ridge on all features 0.76, random forest 0.82, LightGBM 0.83. A 40-config tuning search gained about +0.01, inside the noise. Most of the signal was square footage and property type.
3. **ZIP sensitivity.** Neighborhood inputs were per-ZIP medians from ~6 training sales per ZIP. The same house's estimate moved about 10% (90th percentile ~34%) when its location matched the neighboring ZIP. Shrinking ZIP statistics toward the county (weight n/(n+20), chosen by paired CV) cut that to about 9% (27%) without hurting accuracy, but for the dashboard's default map position the gap between the two nearest ZIPs grew from 12% to 18%.
4. **Unverifiable location matching.** Coordinates were ZIP centers plus random jitter, so matching a map pin to the nearest ZIP "worked" for 192 of 198 test homes only for circular reasons. There was no way to measure how well it would do on a real address.

The conclusion was that a point estimate for an individual home implied more precision than ~1,000 sales and ZIP-level location can support, so the predictor was removed. The complete code, tests and reports are in git history (commit `13a8747`).
