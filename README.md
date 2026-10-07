# 🏠 Miami Real Estate Price Predictor

A machine learning pipeline that predicts residential property prices in Miami-Dade and Broward counties using geospatial features, neighborhood aggregations, and gradient boosting, with SHAP-based feature attribution and an interactive Streamlit dashboard.

**Live Demo**: [miami-real-estate-ml.onrender.com](https://miami-real-estate-ml.onrender.com/) (free tier, so the first load may take a moment to wake up)

**Run locally**: `streamlit run app.py` (opens at `http://localhost:8501`)

---

## 🎯 Project Overview

This project demonstrates end-to-end applied ML: data acquisition → feature engineering → model training → explainability → deployment. It combines:

- **990 Miami-Dade & Broward property sales** (2026 Kaggle dataset, filtered by county via ZIP-code geocoding)
- **20 engineered features** (property, geospatial, neighborhood, temporal)
- **LightGBM regressor** with R² = 0.992 and 4.6% mean price error
- **SHAP explanations** for every prediction
- **Interactive Streamlit dashboard** for price estimation, a neighborhood map, and model diagnostics

---

## 📊 Key Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **R² Score** | 0.9921 | ≥ 0.80 | ✅ |
| **RMSE (log scale)** | 0.0674 | ≤ 0.20 | ✅ |
| **MAE (log scale)** | 0.0447 | ≤ 0.18 | ✅ |
| **Mean Price Error** | 4.6% | ≤ 20% | ✅ |

**Top 3 Features** (by model gain):
1. `price_per_sqft` — Market signal & location premium
2. `sqft` — Property size
3. `sqft_log` — Log-scaled size (non-linear effects)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip or conda

### Installation

```bash
# Clone repo
git clone https://github.com/brianravelo28/miami-real-estate-ml.git
cd miami-real-estate-ml

# Install dependencies
pip install -r requirements.txt

# Download dataset
# Go to: https://www.kaggle.com/datasets/kanchana1990/florida-real-estate-sold-dataset-2026
# Download CSV → place at: data/florida_sold_2026.csv
```

### Run Full Pipeline

```bash
# Execute the pipeline steps in sequence
python run_pipeline.py
```

Or run individual steps:

```bash
python src/01_load_data.py              # Load, geocode & validate
python src/02_engineer_features.py      # Feature creation
python src/03_train_model.py            # Model training
python src/04_shap_analysis.py          # Explainability
```

Then launch the dashboard:

```bash
streamlit run app.py                    # http://localhost:8501
```

---

## 📁 Project Structure

```
miami-real-estate-ml/
├── README.md                          # This file
├── LICENSE                            # MIT license
├── app.py                             # Streamlit dashboard (entry point)
├── requirements.txt                   # Python dependencies
├── render.yaml, Dockerfile            # Deployment config
├── .gitignore                         # Git ignore rules
├── run_pipeline.py                    # Master executor
│
├── src/
│   ├── 01_load_data.py                # Step 1: Load, geocode & validate data
│   ├── 02_engineer_features.py        # Step 2: Feature engineering
│   ├── 03_train_model.py              # Step 3: Train LightGBM
│   ├── 04_shap_analysis.py            # Step 4: SHAP explanations
│   ├── 05_build_dashboard.py          # Legacy Dash dashboard (superseded by app.py)
│   └── real_estate_config.py          # Single source of truth (paths, params)
│
├── data/
│   ├── acquisition.md                 # How to download dataset
│   ├── florida_sold_2026.csv          # Raw Kaggle data (downloaded manually)
│   ├── features_engineered.pkl        # Output of Step 1
│   ├── X_train.pkl, y_train.pkl       # Output of Step 2
│   ├── X_test.pkl, y_test.pkl         # Output of Step 2
│
├── models/
│   └── lightgbm_miami_v1.pkl          # Trained model (Step 3 output)
│
├── reports/
│   ├── evaluation_report.txt          # Model performance metrics
│   ├── residuals.png                  # Residual plots
│   ├── actual_vs_predicted.png        # Predictions vs reality
│   ├── shap_summary.png               # SHAP summary plot
│   ├── shap_feature_importance.png    # Feature importance (SHAP)
│   ├── shap_force_sample_*.png        # Individual prediction explanations
│   ├── shap_dependence_plots.png      # Feature dependence analysis
│   └── SHAP_INTERPRETATION_GUIDE.md   # How to read SHAP plots
│
└── docs/
    ├── ARCHITECTURE.md                # System design & data flow
    ├── METHODOLOGY.md                 # ML approach & trade-offs
    ├── DATA_SCHEMA.md                 # Feature definitions & transformations
    └── DEPLOYMENT.md                  # Render deployment guide
```

---

## 🔄 Data Flow

```
Florida Real Estate CSV (Kaggle)
    ↓ (Step 1: Load & Validate)
Cleaned dataset: 990 Miami-Dade & Broward properties
    ↓ (Step 2: Feature Engineering)
20 engineered features + train/test split (80/20)
    ↓ (Step 3: Train Model)
LightGBM model (200 boosting rounds)
    ├─→ (Step 4: SHAP Analysis) → Explainability plots
    └─→ Streamlit dashboard (app.py) → Interactive predictions
```

---

## 📊 Dashboard Features

The Streamlit app (`streamlit run app.py`) provides three tabs:

### 🔍 **Tab 1: Search & Predict**
- Input property details (beds, baths, sqft, property type)
- Pick a location by panning a map under a fixed crosshair (active within Miami-Dade/Broward)
- Get a price prediction with a confidence range
- View a SHAP bar chart of the features driving that prediction

### 🗺️ **Tab 2: Neighborhood Map**
- Interactive Folium map of predicted prices across Miami-Dade & Broward
- Median, mean, and range stats that update to the properties in the current map view

### 📈 **Tab 3: Model Diagnostics**
- R² score, RMSE, MAE breakdown
- Residual plot (predictions vs actual)
- Global feature importance (mean absolute SHAP value)

---

## 🛠️ Configuration

All parameters are centralized in [`real_estate_config.py`](src/real_estate_config.py):

```python
# Data filtering
MIN_PRICE = 50_000
MAX_PRICE = 10_000_000
MIN_SQFT = 400
MAX_SQFT = 10_000

# Model hyperparameters
LIGHTGBM_NUM_ROUNDS = 200
LIGHTGBM_LEARNING_RATE = 0.05

# Coverage area (filtered by county via ZIP geocoding)
TARGET_COUNTIES = ['Miami-Dade', 'Broward']

# Geospatial reference points
DOWNTOWN_MIAMI = (25.7617, -80.1918)
BRICKELL = (25.7582, -80.1911)
MIAMI_BEACH = (25.7945, -80.1298)
```

To adjust parameters, edit this file once—all scripts automatically pick up changes.

---

## 🤖 Model Details

### Architecture
- **Algorithm**: LightGBM (Gradient Boosting Decision Trees)
- **Target**: Log-transformed sale price (for reduced skewness)
- **Features**: 20 numeric features (no categorical encoding needed)
- **Train/Test Split**: 80/20, stratified by neighborhood price tier

### Feature Categories

| Category | Examples | Count |
|----------|----------|-------|
| **Property** | beds, baths, sqft, property_age, is_condo | 6 |
| **Geospatial** | lat, lon, dist_downtown, dist_brickell, near_coast | 5 |
| **Neighborhood** | neighborhood_median_price, neighborhood_price_std | 3 |
| **Temporal** | sale_year, sale_month | 2 |
| **Risk** | flood_risk_percentile | 1 |
| **Interaction** | price_per_sqft, list_to_sold_ratio | 2 |

### Hyperparameters
```python
{
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
}
```

---

## 📈 Explainability

All predictions include **SHAP (SHapley Additive exPlanations)** values:

- **Summary Plot**: Shows which features push prices up/down globally
- **Force Plot**: For each prediction, displays individual feature contributions
- **Dependence Plot**: Scatter plot of feature value vs SHAP value

Example: "This property is predicted at $750K because price_per_sqft is high (+$200K relative to baseline), but older property_age lowers it (-$50K)."

---

## 🔬 Model Limitations & Next Steps

### Current Limitations
- Trained only on Miami-Dade & Broward 2026 data; may not generalize to other areas/years
- Geospatial features use ZIP-code centroids (plus small jitter), not exact property addresses
- No flood risk from FEMA shapefiles (using latitude proxy instead)
- No macroeconomic features (interest rates, inventory, etc.)

### Potential Improvements
1. **Add FEMA shapefiles** for true flood zone encoding
2. **Temporal dynamics**: Include YoY price trends, interest rates
3. **Ensemble**: Combine LightGBM with neural networks for hybrid predictions
4. **Cross-market**: Retrain on NYC, LA, Chicago for transfer learning
5. **Production deployment**: FastAPI backend + React frontend + PostgreSQL

---

## 📦 Dependencies

Core libraries (see `requirements.txt` for versions):
- `pandas`, `numpy` — Data manipulation
- `scikit-learn` — ML utilities (train/test split, metrics)
- `lightgbm` — Gradient boosting
- `shap` — Model explainability
- `streamlit`, `plotly` — Dashboard and interactive charts
- `folium`, `streamlit-folium` — Interactive maps
- `geopy`, `pgeocode` — Distance calculations and ZIP-code geocoding
- `pillow` — Image handling

---

## 📜 License

MIT License — freely use, modify, and distribute for commercial or personal projects. See [`LICENSE`](LICENSE) for details.

---

## 🤝 Contributing

Found a bug or want to improve the model? Pull requests welcome!

**Process**:
1. Fork the repo
2. Create a feature branch (`git checkout -b feature/improve-features`)
3. Commit changes (`git commit -m "Add flood zone encoding"`)
4. Push to branch (`git push origin feature/improve-features`)
5. Open a Pull Request

---

## 📞 Attribution

**Author**: [brianravelo28](https://github.com/brianravelo28)  
**Dataset**: [Florida Real Estate Sold 2026](https://www.kaggle.com/datasets/kanchana1990/florida-real-estate-sold-dataset-2026) by Kanchana Ranasinghe

---

## 🎓 Resources & References

- [SHAP Documentation](https://shap.readthedocs.io/) — Model interpretability
- [LightGBM Best Practices](https://lightgbm.readthedocs.io/en/latest/Features.html) — Hyperparameter tuning
- [Streamlit Docs](https://docs.streamlit.io/) — Dashboard framework
- [Geospatial Feature Engineering](https://towardsdatascience.com/spatial-machine-learning-4c7d86920d56) — Location-based ML

---

**Last Updated**: October 2026
