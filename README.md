---
title: Miami Real Estate Price Predictor
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# 🏠 Miami Real Estate Price Predictor

A production-ready machine learning pipeline that predicts residential property prices in Miami using geospatial features, neighborhood aggregations, and gradient boosting. Features explainable predictions with SHAP values and an interactive web dashboard.

**Live Demo**: Interactive dashboard at `http://localhost:7860` after running the pipeline.

---

## 🎯 Project Overview

This project demonstrates end-to-end applied ML: data acquisition → feature engineering → model training → explainability → deployment. It combines:

- **1,068 Miami property sales** (2026 Kaggle dataset)
- **20 engineered features** (property, geospatial, neighborhood, temporal)
- **LightGBM regressor** with R² = 0.987 and 6.1% mean price error
- **SHAP explanations** for every prediction
- **Interactive Dash dashboard** for price estimation and model diagnostics

---

## 📊 Key Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **R² Score** | 0.9874 | ≥ 0.80 | ✅ |
| **RMSE (log scale)** | 0.1007 | ≤ 0.20 | ✅ |
| **MAE (log scale)** | 0.0596 | ≤ 0.18 | ✅ |
| **Mean Price Error** | 6.1% | ≤ 20% | ✅ |

**Top 3 Features** (by SHAP importance):
1. `price_per_sqft` — Market signal & location premium
2. `sqft` — Property size (non-linear effects via log transform)
3. `neighborhood_median_price` — ZIP-level market tier

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

```bash
# Clone repo
git clone https://github.com/yourusername/miami-real-estate-ml.git
cd miami-real-estate-ml

# Install dependencies
pip install -r requirements.txt

# Download dataset
# Go to: https://www.kaggle.com/datasets/kanchana1990/florida-real-estate-sold-dataset-2026
# Download CSV → place at: data/florida_sold_2026.csv
```

### Run Full Pipeline

```bash
# Execute all 5 steps in sequence
python run_pipeline.py
```

Or run individual steps:

```bash
python src/01_load_data.py              # Load & validate
python src/02_engineer_features.py      # Feature creation
python src/03_train_model.py            # Model training
python src/04_shap_analysis.py          # Explainability
python src/05_build_dashboard.py        # Web app (http://localhost:7860)
```

---

## 📁 Project Structure

```
miami-real-estate-ml/
├── README.md                          # This file
├── LICENSE                            # MIT license
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── run_pipeline.py                    # Master executor
│
├── src/
│   ├── 01_load_data.py                # Step 1: Load & validate data
│   ├── 02_engineer_features.py        # Step 2: Feature engineering
│   ├── 03_train_model.py              # Step 3: Train LightGBM
│   ├── 04_shap_analysis.py            # Step 4: SHAP explanations
│   ├── 05_build_dashboard.py          # Step 5: Interactive dashboard
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
├── notebooks/
│   ├── 01_EDA.ipynb                   # Exploratory data analysis
│   └── 02_Feature_Engineering_Deep_Dive.ipynb  # Feature creation rationale
│
└── docs/
    ├── ARCHITECTURE.md                # System design & data flow
    ├── METHODOLOGY.md                 # ML approach & trade-offs
    └── DATA_SCHEMA.md                 # Feature definitions & transformations
```

---

## 🔄 Data Flow

```
Florida Real Estate CSV (Kaggle)
    ↓ (Step 1: Load & Validate)
Cleaned dataset: 1,068 Miami properties
    ↓ (Step 2: Feature Engineering)
20 engineered features + train/test split (80/20)
    ↓ (Step 3: Train Model)
LightGBM model (200 boosting rounds)
    ├─→ (Step 4: SHAP Analysis) → Explainability plots
    └─→ (Step 5: Dashboard) → Interactive predictions
```

---

## 📊 Dashboard Features

The interactive Dash app (`localhost:7860`) provides three tabs:

### 🔍 **Tab 1: Search & Predict**
- Input property details (beds, baths, sqft, location)
- Get instant price prediction + confidence band
- View SHAP force plot explaining the prediction
- See "similar sales" from training data

### 🗺️ **Tab 2: Neighborhood Map**
- Interactive Folium heatmap of Miami
- Price distribution by ZIP code
- Hover for neighborhood statistics

### 📈 **Tab 3: Model Diagnostics**
- R² score, RMSE, MAE breakdown
- Residual plot (predictions vs actual)
- SHAP summary plot (feature importance)
- Top 10 features by gain

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
- Trained only on Miami 2026 data; may not generalize to other cities/years
- Geospatial features use synthetic lat/lon (based on ZIP codes)
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
- `plotly`, `dash` — Interactive visualizations
- `geopy` — Distance calculations
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

## 📞 Contact & Attribution

**Author**: Your Name  
**Email**: your.email@example.com  
**Dataset**: [Florida Real Estate Sold 2026](https://www.kaggle.com/datasets/kanchana1990/florida-real-estate-sold-dataset-2026) by Kanchana Ranasinghe

---

## 🎓 Resources & References

- [SHAP Documentation](https://shap.readthedocs.io/) — Model interpretability
- [LightGBM Best Practices](https://lightgbm.readthedocs.io/en/latest/Features.html) — Hyperparameter tuning
- [Dash by Plotly](https://dash.plotly.com/) — Interactive web apps
- [Geospatial Feature Engineering](https://towardsdatascience.com/spatial-machine-learning-4c7d86920d56) — Location-based ML

---

**Last Updated**: August 2026  
**Status**: Production-ready ✅
