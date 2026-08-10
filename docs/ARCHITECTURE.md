# System Architecture

## 🏗️ Overview

This document describes the end-to-end architecture of the Miami Real Estate ML pipeline, from data ingestion to interactive predictions.

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Kaggle Dataset (10,893 records)                      │
│                  florida_real_estate_sold_dataset_2026                   │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  STEP 1: LOAD   │
                    │  & VALIDATE     │
                    └────────┬────────┘
                             │
           ┌─────────────────┴──────────────────┐
           │ - Filter to Miami metro (ZIP 331,334)
           │ - Remove price/sqft outliers
           │ - Generate synthetic lat/lon
           │ - Fill missing values
           │
           └─────────────────┬──────────────────┘
                             │
                    ┌────────▼──────────┐
                    │  Cleaned Data     │
                    │ (1,068 records)   │
                    │ features_eng.pkl  │
                    └────────┬──────────┘
                             │
                    ┌────────▼──────────────┐
                    │  STEP 2: FEATURE     │
                    │  ENGINEERING         │
                    └────────┬──────────────┘
                             │
    ┌────────────────────────┼────────────────────────┐
    │                        │                        │
    ▼                        ▼                        ▼
Property Features    Geospatial Features    Neighborhood Agg
- beds, baths        - lat, lon             - median_price
- sqft_log           - dist_downtown        - price_std
- property_age       - dist_brickell        - sales_count
- price_per_sqft     - dist_beach           - price_tier
- is_condo           - near_coast           
- is_townhouse                              Temporal Features
- property_age_ord.  Flood Risk             - sale_year
- list_to_sold_ratio - flood_risk_pctl     - sale_month
                                             
    │                        │                        │
    └────────────────────────┼────────────────────────┘
                             │
                    ┌────────▼──────────────┐
                    │  20-Feature Matrix   │
                    │  + train/test split  │
                    │  (80/20, stratified) │
                    └────────┬──────────────┘
                             │
                    ┌────────▼──────────────┐
                    │  STEP 3: TRAIN       │
                    │  LIGHTGBM MODEL      │
                    └────────┬──────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   Evaluation          Model Artifacts      Visualizations
   - R² = 0.987        - model.pkl          - residuals.png
   - MAE = 6.1%        - feature_import     - actual_vs_pred.png
   - RMSE = 0.1007                          
                                             
                    ┌────────▼──────────────┐
                    │  STEP 4: SHAP        │
                    │  EXPLAINABILITY      │
                    └────────┬──────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   SHAP Values      Force Plots          Feature Importance
   (214, 20)        - 5 samples          - summary_plot.png
   Expected: 13.3   - individual explns  - dependence_plots.png
                                          
                    ┌────────▼──────────────┐
                    │  STEP 5: DASHBOARD   │
                    │  Dash Web App        │
                    └────────┬──────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   Tab 1:            Tab 2:              Tab 3:
   Search & Predict  Neighborhood Map   Model Diagnostics
   - Input property  - Folium heatmap   - R² score
   - Get prediction  - ZIP-level stats  - Residual plot
   - SHAP force plot - Price dist.      - SHAP summary
                     - Market trends    - Top 10 features
                                         
                             │
                    ┌────────▼──────────────┐
                    │  Interactive Web UI   │
                    │  localhost:7860       │
                    └───────────────────────┘
```

---

## Component Breakdown

### 1. **Data Ingestion (Step 1)**

**File**: `src/01_load_data.py`

**Inputs**:
- Raw CSV from Kaggle (10,893 Florida property sales)
- Config parameters from `real_estate_config.py`

**Processing**:
1. Load CSV → pandas DataFrame
2. Filter by ZIP code prefixes (Miami metro: 331, 334)
3. Validate required columns: zip, beds, baths, sqft, yr_built, lastSoldPrice
4. Remove outliers:
   - Price: $50K–$10M
   - Sqft: 400–10,000
5. Handle missing values (drop critical columns)
6. Generate synthetic lat/lon from ZIP codes (since dataset lacks coordinates)
7. Add default flood_risk = 'X' (no FEMA data available)

**Outputs**:
- `data/features_engineered.pkl` — Cleaned DataFrame (1,068 records, 17 columns)

**Why pickle?**
- Preserves data types (no re-casting on reload)
- Fast I/O for large datasets
- Seamless pandas integration

---

### 2. **Feature Engineering (Step 2)**

**File**: `src/02_engineer_features.py`

**Inputs**:
- Cleaned data from Step 1
- Config: reference points, bins, feature list

**Feature Categories**:

#### Property Features (6 features)
- `sqft_log` = log(sqft) — non-linear size effect
- `property_age` = 2026 - yr_built — newer = premium
- `price_per_sqft` = lastSoldPrice / sqft — market density
- `list_to_sold_ratio` = lastSoldPrice / listPrice — negotiation power
- `is_condo` = 1 if type=='condo' else 0 — property classification
- `is_townhouse` = 1 if type=='townhouse' else 0
- `property_age_ordinal` = binned(property_age, bins=[0,5,20,40,200])

#### Geospatial Features (5 features)
- `lat`, `lon` — raw coordinates (synthetic, but spatially meaningful)
- `dist_downtown` = haversine(lat, lon, (25.7617, -80.1918)) — miles to CBD
- `dist_brickell` = haversine(..., (25.7582, -80.1911))
- `dist_beach` = haversine(..., (25.7945, -80.1298)) — waterfront proxy
- `near_coast` = 1 if lat > 25.8 else 0 — latitude as coast proximity

#### Neighborhood Aggregations (3 features)
Grouped by ZIP code:
- `neighborhood_median_price` = median(lastSoldPrice) per ZIP
- `neighborhood_price_std` = std(lastSoldPrice) per ZIP
- `neighborhood_sales_count` = count() per ZIP
- `neighborhood_price_tier` = ordinal(neighborhood_median_price) — used for stratification only

#### Temporal Features (2 features)
- `sale_year` = 2026 (default, no saleDate in data)
- `sale_month` = 6 (mid-year default)

#### Risk Features (1 feature)
- `flood_risk_percentile` = rank(25.8 - lat) / n — latitude-based proxy
  (Lower latitude = closer to coast = higher flood risk)

**Feature Selection**:
- Start with all 20 created features
- Select subset from `FEATURE_COLS` (config)
- Drop features with >50% missing (none in this case)

**Train/Test Split**:
- 80/20 split (854 train, 214 test)
- Stratified by `neighborhood_price_tier` (ensures all price tiers in both sets)
- `random_state=42` for reproducibility

**Outputs**:
- `data/X_train.pkl` — (854, 20) feature matrix
- `data/y_train.pkl` — (854,) log-transformed target
- `data/X_test.pkl` — (214, 20) feature matrix
- `data/y_test.pkl` — (214,) log-transformed target

---

### 3. **Model Training (Step 3)**

**File**: `src/03_train_model.py`

**Algorithm**: LightGBM (Gradient Boosting Decision Trees)

**Why LightGBM?**
- Fast training (handles 1,068 records in seconds)
- Interpretable feature importance via gain/split/cover
- Native SHAP support
- Handles numeric features without preprocessing
- Robust to outliers (tree splits are threshold-based)

**Target Transformation**:
- `y = log1p(lastSoldPrice)` — reduces right skewness
- Interpretation: RMSE in log scale → ≈ percentage error in original price

**Hyperparameters** (from config):
```python
{
    'objective': 'regression',      # Predict continuous values
    'metric': 'rmse',               # Optimize root mean squared error
    'num_leaves': 31,               # Tree depth (2^5 = 32 max leaves)
    'learning_rate': 0.05,          # Shrinkage (prevents overfitting)
    'feature_fraction': 0.8,        # Random subsampling of features
    'bagging_fraction': 0.8,        # Random subsampling of samples
    'bagging_freq': 5,              # Bagging every 5 boosting rounds
    'verbose': -1,                  # Suppress logs
}
num_boost_round=200                 # 200 decision trees
```

**Training Process**:
1. Create LightGBM Dataset from (X_train, y_train)
2. Train 200 rounds of boosting
3. Log every 50 rounds (progress indicator)
4. Evaluate on test set

**Evaluation Metrics** (on test set):
```
y_pred = model.predict(X_test)

RMSE = sqrt(mean((y_true - y_pred)^2))
       = 0.1007 (log scale)
       ≈ exp(0.1007) - 1 = 10.6% price error

MAE = mean(|y_true - y_pred|)
    = 0.0596 (log scale)
    ≈ 6.1% average price error (more interpretable)

R² = 1 - (SS_res / SS_tot)
   = 0.9874 (excellent fit)
```

**Feature Importance** (via gain = reduction in loss per feature):
1. `price_per_sqft` — 3,693 (most splits on this feature)
2. `sqft` — 1,370
3. `sqft_log` — 326 (multicollinearity with sqft, but improves splits)
4. [... 17 more features]

**Outputs**:
- `models/lightgbm_miami_v1.pkl` — Trained model (binary serialized)
- `reports/evaluation_report.txt` — Metrics summary
- `reports/residuals.png` — Residual plot (errors vs predictions)
- `reports/actual_vs_predicted.png` — Prediction accuracy plot

---

### 4. **Model Explainability (Step 4)**

**File**: `src/04_shap_analysis.py`

**SHAP (SHapley Additive exPlanations)**:
- Game-theoretic approach to model interpretability
- Assigns each feature a value representing its contribution to each prediction
- Satisfies 4 desirable properties (symmetry, dummy, additivity, efficiency)

**Implementation**:
```python
import shap

explainer = shap.TreeExplainer(model)  # Optimized for tree models
shap_values = explainer.shap_values(X_test)  # (214, 20) array
base_value = explainer.expected_value  # ≈ mean(y_train) = 13.295
```

**Outputs**:

| Artifact | Purpose |
|----------|---------|
| `shap_summary.png` | Dot plot showing feature importance & direction |
| `shap_feature_importance.png` | Bar chart of mean absolute SHAP values |
| `shap_force_sample_1-5.png` | Individual predictions explained |
| `shap_dependence_plots.png` | Scatter plots: feature value vs SHAP value |
| `SHAP_INTERPRETATION_GUIDE.md` | User-friendly documentation |

**Example Reading a Force Plot**:
```
Base value: $500K (model's average prediction)
  ├─ price_per_sqft=120 [red] +$150K (above average market rate)
  ├─ sqft=2,000 [red] +$120K (larger than median)
  ├─ property_age=8 [blue] -$30K (slightly old)
  └─ ... other features ...
Final prediction: $740K
```

---

### 5. **Interactive Dashboard (Step 5)**

**File**: `src/05_build_dashboard.py`

**Framework**: Dash (Plotly) + Flask

**Architecture**:
```
User's Browser
     │
     └─→ http://localhost:7860
              │
         ┌────▼─────┐
         │ Dash App │ (Flask server)
         └────┬─────┘
              │
         ┌────▼───────────────┐
         │ Loaded Artifacts   │
         ├─ model.pkl         │
         ├─ X_test.pkl        │
         ├─ SHAP values       │
         ├─ Images (.png)     │
         └────┬───────────────┘
              │
    ┌─────────┼─────────┐
    │         │         │
    ▼         ▼         ▼
  Tab 1:   Tab 2:    Tab 3:
  Predict  Heatmap   Diagnostics
```

**Tab 1: Search & Predict**
- User inputs: beds, baths, sqft, location (ZIP or lat/lon)
- Backend:
  1. Create feature row (apply same transformations as training)
  2. `y_pred_log = model.predict([feature_row])`
  3. `price = exp(y_pred_log) - 1` (convert back from log scale)
  4. `shap_val = explainer.shap_values([feature_row])[0]` (explain prediction)
  5. Generate force plot + similar sales
- Display: Predicted price, SHAP force plot, confidence interval

**Tab 2: Neighborhood Map**
- Folium map of Miami
- Heatmap layer: ZIP code boundaries colored by median price
- Markers: Sample sales with price on hover
- Interactivity: Click to see ZIP stats

**Tab 3: Model Diagnostics**
- KPI cards: R², RMSE, MAE
- Residual plot: (y_pred vs y_true - y_pred)
- SHAP summary plot
- Feature importance bar chart

**Callbacks** (Dash reactivity):
```python
@callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('beds-input', 'value'),
    State('baths-input', 'value'),
    # ... more states
)
def update_prediction(n_clicks, beds, baths, ...):
    # Compute & return prediction
```

---

## Deployment Considerations

### Current State (Development)
- Local execution (`python 05_build_dashboard.py`)
- Single-threaded Dash server
- In-memory model loading

### Production Readiness
To deploy to cloud (AWS, GCP, Heroku):

1. **Containerize**:
   ```dockerfile
   FROM python:3.9
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   CMD ["python", "05_build_dashboard.py"]
   ```

2. **Use production ASGI server**:
   ```bash
   gunicorn --workers 4 --timeout 120 \
            --bind 0.0.0.0:8000 \
            'app:server'  # Dash's underlying Flask app
   ```

3. **Environment variables**:
   ```bash
   export MODEL_PATH=/data/models/lightgbm_miami_v1.pkl
   export DASH_PORT=8000
   export LOG_LEVEL=INFO
   ```

4. **Database** (if caching predictions):
   - PostgreSQL for prediction history
   - Redis for SHAP value caching (expensive to recompute)

---

## Error Handling & Validation

### Pipeline Robustness

| Step | Failure Mode | Mitigation |
|------|--------------|-----------|
| **Load** | Missing CSV | Instructions to download from Kaggle |
| **Load** | Bad columns | Validate required columns upfront |
| **Engineer** | NaN in features | Fillna with median/mode |
| **Train** | Categorical types | Type-check before passing to LightGBM |
| **SHAP** | Slow computation | Sample test set if >10K rows |
| **Dashboard** | Image not found | Try/except + fallback placeholder |

### Data Quality Checks
```python
# Outlier detection & removal
price_ok = (df['lastSoldPrice'] >= MIN_PRICE) & 
           (df['lastSoldPrice'] <= MAX_PRICE)

sqft_ok = (df['sqft'] >= MIN_SQFT) & (df['sqft'] <= MAX_SQFT)

# Missing values
critical_cols = ['beds', 'baths', 'sqft', 'lastSoldPrice']
df = df.dropna(subset=critical_cols)
```

---

## Performance Metrics

| Operation | Time | Notes |
|-----------|------|-------|
| **Step 1** (Load + Validate) | 1-2 sec | CSV read + filtering |
| **Step 2** (Feature Engineering) | 5-10 sec | Haversine distance, groupby aggregations |
| **Step 3** (Train LightGBM) | 10-20 sec | 200 boosting rounds, 854 samples |
| **Step 4** (SHAP Analysis) | 30-60 sec | TreeExplainer on 214 test samples |
| **Step 5** (Dashboard Load) | 2-5 sec | Model + images in memory |

**Total Pipeline Runtime**: ≈2–3 minutes (first run)

---

## Future Enhancements

1. **Incremental Learning**: Retrain model weekly on new sales
2. **A/B Testing**: Serve multiple model versions to subset of users
3. **Confidence Intervals**: Use quantile regression for prediction bounds
4. **API Layer**: FastAPI backend for mobile app integration
5. **Monitoring**: Log predictions, track prediction vs actual prices, detect model drift
