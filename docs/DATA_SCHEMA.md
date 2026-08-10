# Data Schema & Feature Definitions

## 📋 Raw Dataset Schema

**Source**: Kaggle — Florida Real Estate Sold Dataset 2026  
**Records**: 10,893 (after filtering to Miami metro: 1,068)

| Column | Type | Null % | Description |
|--------|------|--------|-------------|
| `type` | string | 0.0% | Property type (Apartment, Single Family, etc.) |
| `sub_type` | string | 0.5% | Detailed type (Condo, Townhouse, etc.) |
| `listPrice` | float | 0.0% | Listed price (USD) |
| `lastSoldPrice` | int | 0.0% | Final sale price (USD) — **TARGET** |
| `sqft` | float | 2.0% | Living area (square feet) |
| `stories` | float | 10.2% | Number of stories |
| `beds` | float | 2.5% | Number of bedrooms |
| `baths` | float | 2.4% | Number of bathrooms |
| `baths_full` | float | 15.3% | Full bathrooms (redundant with baths) |
| `baths_full_calc` | float | 18.9% | Calculated full baths (redundant) |
| `garage` | float | 45.2% | Garage spaces (sparse) |
| `year_built` | float | 1.2% | Construction year — **Renamed to `yr_built`** |
| `zip` | string | 0.0% | ZIP code |
| `sanitized_text` | string | 0.0% | Description (unused) |

---

## 🛠️ Feature Engineering Pipeline

### Input → Output Transformation

**Raw Data**:
```
type, sub_type, listPrice, lastSoldPrice, sqft, beds, baths, 
year_built, zip, lat (synthetic), lon (synthetic), flood_risk
```

**After Step 2 (Feature Engineering)**:
```
20 numeric features + target (log-transformed price)
```

---

## 📊 Engineered Features (20 Total)

### 1. Property Features (6 features)

#### `beds` (numeric: 0–8)
- **Source**: Raw column
- **Type**: Float
- **Range**: 0–8 (mode=3)
- **Missing**: 2.5% (filled with mode=3)
- **Usage**: Linear effect on price; tree splits on values 2, 3, 4
- **SHAP Rank**: #10

#### `baths` (numeric: 0–5+)
- **Source**: Raw column
- **Type**: Float
- **Range**: 0–5 (mode=2)
- **Missing**: 2.4% (filled with mode=2)
- **Usage**: Similar to beds; slight diminishing returns
- **SHAP Rank**: #5

#### `sqft` (numeric: 400–10,000)
- **Source**: Raw column (filtered outliers)
- **Type**: Float
- **Range**: 440–8,546 (median=1,562)
- **Missing**: 2.0% (filled with median=1,562)
- **Usage**: Primary size metric; strong predictor
- **Transform**: Both raw + log included
- **SHAP Rank**: #2

#### `sqft_log` (numeric: 6.1–9.1)
- **Source**: log1p(sqft)
- **Type**: Float
- **Computation**: np.log1p(sqft)
- **Rationale**: Capture non-linear size effects (large homes have diminishing returns)
- **Range**: log1p(440) ≈ 6.1 to log1p(8,546) ≈ 9.1
- **Multicollinearity**: Correlated with sqft, but adds predictive value
- **SHAP Rank**: #3

#### `property_age` (numeric: 0–170 years)
- **Source**: 2026 - year_built
- **Type**: Float
- **Range**: 2026 - 1856 = 0–170 (median ≈ 35 years)
- **Interpretation**: Newer properties command premium; older may need updates
- **Outliers**: Very old properties (pre-1900) rare in Miami
- **SHAP Rank**: #9

#### `property_age_ordinal` (ordinal: 0–3)
- **Source**: Binned property_age
- **Type**: Int (0=new, 1=recent, 2=established, 3=old)
- **Bins**: [0, 5, 20, 40, 200] years
- **Values**:
  - 0: 0–5 years (new construction)
  - 1: 5–20 years (recent)
  - 2: 20–40 years (established)
  - 3: 40+ years (old)
- **Rationale**: Captures non-linear age effects
- **Usage**: LightGBM can use for discrete splits

#### `price_per_sqft` (numeric: $51–$2,340/sqft)
- **Source**: lastSoldPrice / sqft
- **Type**: Float
- **Range**: 51–2,340 (median ≈ 350)
- **Rationale**: Market density metric; encodes location premium
- **Example**: $500K home on 2,000 sqft = $250/sqft (mainstream)
           $500K home on 1,000 sqft = $500/sqft (premium location)
- **SHAP Rank**: #1 (most important feature!)

#### `list_to_sold_ratio` (numeric: 0.5–1.5)
- **Source**: lastSoldPrice / listPrice
- **Type**: Float
- **Computation**: np.clip(lastSoldPrice / listPrice, 0.5, 1.5)
- **Rationale**: Negotiation dynamics (agent success rate)
- **Interpretation**:
  - Ratio < 1.0: Seller lost value (buyer had leverage)
  - Ratio = 1.0: Sold at listing price
  - Ratio > 1.0: Seller gained (bidding war, strong market)
- **Clipping**: Prevents infinities from $0 listings
- **SHAP Rank**: Low (moderate contribution)

#### `is_condo` (binary: 0–1)
- **Source**: sub_type == 'condo' ? 1 : 0
- **Type**: Int
- **Values**: 1 if property is condo, 0 otherwise
- **Distribution**: ≈ 40% condos in Miami metro
- **SHAP Impact**: Typically negative (condos < single-family in price)

#### `is_townhouse` (binary: 0–1)
- **Source**: sub_type == 'townhouse' ? 1 : 0
- **Type**: Int
- **Values**: 1 if townhouse, 0 otherwise
- **Distribution**: ≈ 15% townhouses
- **SHAP Impact**: Typically negative (townhouses < single-family)

---

### 2. Geospatial Features (5 features)

#### `lat` (numeric: 25.7–25.8)
- **Source**: Synthetic (generated from ZIP code)
- **Type**: Float
- **Range**: 25.7269–25.8275
- **Precision**: ±0.01 (≈1 km), added random noise
- **Rationale**: Latitude for distance calculations
- **Limitations**: Synthetic; loses within-ZIP variance
- **SHAP Rank**: Low direct impact (#11+)

#### `lon` (numeric: -80.2 to -80.1)
- **Source**: Synthetic (generated from ZIP code)
- **Type**: Float
- **Range**: -80.2188 to -80.0969
- **Precision**: ±0.01 (≈0.7 km at Miami latitude)
- **Rationale**: Longitude for distance calculations
- **SHAP Rank**: #8

#### `dist_downtown` (numeric: 0–15 miles)
- **Source**: haversine((lat, lon), (25.7617, -80.1918))
- **Type**: Float
- **Reference Point**: Downtown Miami CBD (Brickell)
- **Range**: 0–15 miles (median ≈ 8 miles)
- **Computation**:
  ```python
  from geopy.distance import geodesic
  dist = geodesic((lat, lon), (25.7617, -80.1918)).miles
  ```
- **Interpretation**: Closer to downtown → typically higher price (CBD premium)
- **SHAP Rank**: High (originally #1 in older version; now #7)

#### `dist_brickell` (numeric: 0–20 miles)
- **Source**: haversine((lat, lon), (25.7582, -80.1911))
- **Type**: Float
- **Reference Point**: Brickell (luxury neighborhood, South Miami)
- **Range**: 0–20 miles
- **Interpretation**: Luxury district; distance to premium market
- **SHAP Rank**: #7

#### `dist_beach` (numeric: 0–10 miles)
- **Source**: haversine((lat, lon), (25.7945, -80.1298))
- **Type**: Float
- **Reference Point**: Miami Beach (25.7945, -80.1298)
- **Range**: 0–10 miles
- **Interpretation**: Waterfront proxy; proximity to beach premium
- **SHAP Rank**: Moderate (#15+)

#### `near_coast` (binary: 0–1)
- **Source**: lat > 25.8 ? 1 : 0
- **Type**: Int
- **Values**: 1 if latitude > 25.8 (northern Miami, closer to coast)
- **Rationale**: Simple binary waterfront indicator
- **Limitations**: Imperfect; assumes latitude = coastal proximity
- **Better Approach**: Use actual shoreline distance or flood zone

---

### 3. Neighborhood Aggregation Features (3 features)

Features are ZIP-code level statistics; same value for all properties in same ZIP.

#### `neighborhood_median_price` (numeric: $200K–$3M)
- **Source**: median(lastSoldPrice) per ZIP
- **Type**: Float
- **Computation**:
  ```python
  zip_stats = df.groupby('zip')['lastSoldPrice'].agg('median')
  df = df.merge(zip_stats, on='zip')
  ```
- **Range**: $200K–$3M (Miami's ZIP median range)
- **Interpretation**: Market tier indicator
  - <$300K: Budget/Affordable
  - $300K–$500K: Mainstream
  - $500K–$1M: Luxury
  - >$1M: Ultra-luxury
- **SHAP Rank**: #6

#### `neighborhood_price_std` (numeric: $50K–$2M)
- **Source**: std(lastSoldPrice) per ZIP
- **Type**: Float
- **Interpretation**: Price variance within neighborhood
  - Low std: Homogeneous, stable market
  - High std: Diverse properties, more negotiation room
- **Missing**: Filled with median std ≈ $500K
- **SHAP Rank**: #4 (surprisingly important!)

#### `neighborhood_sales_count` (numeric: 10–500)
- **Source**: count() per ZIP
- **Type**: Int
- **Range**: Minimum 10 (outlier filter), maximum 500+
- **Interpretation**: Market liquidity
  - High count: Liquid market, comparable sales available
  - Low count: Illiquid, hard to price
- **SHAP Rank**: Low (#20+)

---

### 4. Temporal Features (2 features)

#### `sale_year` (numeric: 2026)
- **Source**: saleDate.year or default 2026
- **Type**: Int
- **Value**: 2026 (constant; no temporal variation)
- **Opportunity**: If historical data collected, capture multi-year trends
- **SHAP Rank**: Very low (no variance)

#### `sale_month` (numeric: 1–12, default 6)
- **Source**: saleDate.month or default 6 (June)
- **Type**: Int
- **Value**: 6 (constant; no monthly seasonality)
- **Opportunity**: If date data available, capture seasonal patterns
  - Summer (Jun–Aug): Higher prices (buyer motivation)
  - Winter (Dec–Feb): Lower prices (fewer buyers)
- **SHAP Rank**: Very low (no variance)

---

### 5. Risk Features (1 feature)

#### `flood_risk_percentile` (numeric: 0–1)
- **Source**: rank(25.8 - lat) / n
- **Type**: Float
- **Range**: 0–1 (percentile scale)
- **Computation**:
  ```python
  df['flood_risk_proxy'] = (25.8 - df['lat']).clip(0, None)
  df['flood_risk_percentile'] = df['flood_risk_proxy'].rank(pct=True)
  ```
- **Interpretation**:
  - 0 = Lowest risk (far south, inland)
  - 1 = Highest risk (close to northern coast)
- **Limitations**: Oversimplified; assumes latitude = flood risk
- **Better Approach**: Use FEMA flood zone shapefiles:
  ```
  zones: X (0, no risk) < A (1) < AE (2) < VE (3, coastal high hazard)
  ```
- **SHAP Rank**: Very low

---

## 📈 Target Variable

### `lastSoldPrice` (Raw)
- **Type**: Int (USD)
- **Range**: $500–$51.7M (after filtering: $50K–$10M)
- **Mean**: $1.18M
- **Median**: $562K
- **Std Dev**: $2.12M
- **Skewness**: 3.2 (highly right-skewed)

### Target Transform: `log1p(lastSoldPrice)`
- **Type**: Float (log scale)
- **Computation**: np.log1p(price) = ln(price + 1)
- **Range**: ln(50K) ≈ 10.8 to ln(10M) ≈ 16.1
- **Mean**: 13.24
- **Std Dev**: 1.08
- **Skewness**: 0.3 (nearly normal!)

**Interpretation**:
- Log scale reduces outlier influence
- RMSE/MAE in log scale ≈ percentage error in original scale
- Example: MAE=0.06 (log) ≈ exp(0.06)−1 = 6.2% price error

---

## 🔄 Data Flow & Pickle Files

### Step 1 Output: `features_engineered.pkl`
```python
import pickle
df = pickle.load(open('data/features_engineered.pkl', 'rb'))
# df.shape: (1068, 17)
# Columns: type, sub_type, listPrice, lastSoldPrice, sqft, 
#          beds, baths, yr_built, zip, lat, lon, flood_risk, ... (17 total)
```

### Step 2 Output: Train/Test Split
```python
X_train = pickle.load(open('data/X_train.pkl', 'rb'))  # (854, 20)
X_test = pickle.load(open('data/X_test.pkl', 'rb'))   # (214, 20)
y_train = pickle.load(open('data/y_train.pkl', 'rb'))  # (854,)
y_test = pickle.load(open('data/y_test.pkl', 'rb'))    # (214,)

# X columns (in order):
# beds, baths, sqft, sqft_log, property_age, price_per_sqft,
# list_to_sold_ratio, is_condo, is_townhouse, lat, lon,
# dist_downtown, dist_brickell, dist_beach, neighborhood_median_price,
# neighborhood_price_std, neighborhood_sales_count, flood_risk_percentile,
# sale_year, sale_month

# y is log-transformed target
# y_actual = np.expm1(y)  # Convert back to original price
```

---

## ✅ Data Quality Checks

### Validation Rules (Applied in Step 1)

| Rule | Threshold | Action |
|------|-----------|--------|
| Price in range | $50K–$10M | Drop if outside |
| Sqft in range | 400–10,000 | Drop if outside |
| Required columns present | beds, baths, sqft, yr_built, lastSoldPrice | Raise error |
| Critical missing values | Any in (beds, baths, sqft, price) | Drop row |
| ZIP code valid | Starts with 331 or 334 (Miami) | Filter to metro |

### Fillna Rules (Applied in Step 2)

| Column | Missing % | Fill Strategy |
|--------|-----------|---|
| `neighborhood_price_std` | Rare | Fill with median std ≈ $500K |
| `list_to_sold_ratio` | Rare | Fill with 1.0 (sold at list) |
| All others | <3% | Already validated in Step 1 |

---

## 📊 Feature Statistics (After Engineering)

```
Feature                       Mean    Std     Min      Max
─────────────────────────────────────────────────────────
beds                         3.14    1.09    0.0      8.0
baths                        2.34    1.01    0.0      5.0
sqft                       1,562    1,195    440    8,546
sqft_log                     7.26    0.62    6.1      9.1
property_age                 34.7    24.3    0.0    170.0
price_per_sqft              360.2    287.1   51.0  2,340.0
list_to_sold_ratio            1.01    0.08   0.5      1.5
is_condo                      0.40    0.49   0.0      1.0
is_townhouse                  0.15    0.36   0.0      1.0
lat                          25.77   0.018  25.7    25.8
lon                         -80.14   0.044 -80.2   -80.1
dist_downtown               8.12    3.85    0.2    15.0
dist_brickell               8.03    3.82    0.1    19.8
dist_beach                   6.24    3.17    0.3    10.0
neighborhood_median_price  592K     457K   200K     3.0M
neighborhood_price_std     518K     312K   100K     1.8M
neighborhood_sales_count   95.4    64.2    10.0   500.0
flood_risk_percentile        0.50    0.29   0.0      1.0
sale_year                   2026    0.0    2026    2026
sale_month                    6.0    0.0     6.0      6.0
```

---

## 🔗 Relationships & Dependencies

### Feature Correlations (Pearson)

High correlations (|r| > 0.7):
- `sqft` ↔ `sqft_log` (r=0.98, expected; same source)
- `beds` ↔ `sqft` (r=0.68, natural)
- `baths` ↔ `sqft` (r=0.71, natural)
- `price_per_sqft` ↔ `lastSoldPrice` (r=0.92, target ingredient!)

Moderate correlations (0.3 < |r| < 0.7):
- `lat` ↔ `dist_downtown` (r=-0.55, latitude captures some location)
- `neighborhood_median_price` ↔ `price_per_sqft` (r=0.52, market tiers)

Low correlations (|r| < 0.3):
- Temporal features (no variance)
- Geospatial distances (mostly independent)

---

## 🚀 Next Steps

### To Improve Data Quality:
1. **Acquire FEMA flood zone shapefiles** for true flood_risk encoding
2. **Collect saleDate** for temporal trends (seasonality, YoY growth)
3. **Geocode coordinates** instead of using ZIP centers (sub-ZIP precision)
4. **Add property condition** (excellent/good/fair/poor → numeric score)
5. **Include HOA fees, taxes** (operating costs, price impact)
6. **Capture amenities** (pool, garage, waterfront, etc.)

### To Monitor Data Drift:
1. Track distribution of features over time (mean, std)
2. Alert if new sales fall outside training range
3. Retrain model quarterly with rolling 12-month window
4. Compare out-of-sample predictions vs actual sales
