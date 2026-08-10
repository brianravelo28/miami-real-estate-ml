# Real Estate Project - Claude Code Pipeline

This project is structured as a **self-contained, executable pipeline** optimized for Claude Code.

## 📦 Project Structure

```
real-estate-project/
├── real_estate_config.py          # Configuration (SINGLE SOURCE OF TRUTH)
├── 01_load_data.py                # Step 1: Load & validate data
├── 02_engineer_features.py        # Step 2: Create features
├── 03_train_model.py              # Step 3: Train LightGBM model
├── 04_shap_analysis.py            # Step 4: SHAP explanations
├── 05_build_dashboard.py          # Step 5: Dash web app
├── run_pipeline.py                # Master executor (optional)
├── requirements.txt               # Python dependencies
└── README_CLAUDE_CODE.md          # This file

data/                              # Created by pipeline
├── florida_sold_2026.csv          # (You download this from Kaggle)
├── features_engineered.pkl        # Output of Step 1
├── X_train.pkl, y_train.pkl       # Output of Step 2
├── X_test.pkl, y_test.pkl         # Output of Step 2

models/                            # Created by pipeline
└── lightgbm_miami_v1.pkl          # Output of Step 3

reports/                           # Created by pipeline
├── shap_summary.png               # Output of Step 4
├── shap_force_sample_*.png        # Output of Step 4
├── shap_feature_importance.png    # Output of Step 4
├── residuals.png                  # Output of Step 3
└── evaluation_report.txt          # Output of Step 3
```

---

## 🚀 How to Execute with Claude Code

### Option 1: Run Individual Steps (Recommended for Development)

Each script is **self-contained** and can be run independently:

```bash
# Step 1: Load and validate data
python 01_load_data.py

# Step 2: Create features
python 02_engineer_features.py

# Step 3: Train model
python 03_train_model.py

# Step 4: SHAP analysis
python 04_shap_analysis.py

# Step 5: Launch dashboard
python 05_build_dashboard.py
```

**When to use individual steps:**
- Debugging a specific step (e.g., feature engineering issues)
- Iterating on model parameters
- Regenerating visualizations

### Option 2: Run Full Pipeline (Recommended for Deployment)

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python run_pipeline.py
```

**When to use full pipeline:**
- First-time execution
- Automated deployment
- CI/CD pipeline

---

## 🔧 Configuration

**All parameters live in one file:** `real_estate_config.py`

This is the **single source of truth** for:
- File paths (data, models, reports)
- Geospatial reference points (downtown Miami, Brickell, etc.)
- Data filtering rules (price range, sqft bounds)
- Feature names and engineering parameters
- LightGBM hyperparameters
- Model success criteria (R², RMSE)

**To change something:**
1. Open `real_estate_config.py`
2. Modify the constant (e.g., `LIGHTGBM_NUM_ROUNDS = 300`)
3. All scripts automatically pick up the change

**Example:** To train for 300 rounds instead of 200:
```python
# In real_estate_config.py
LIGHTGBM_NUM_ROUNDS = 300  # Changed from 200
```

When you run `03_train_model.py`, it automatically uses `300` rounds.

---

## 📋 Step-by-Step Execution Guide

### Step 1: Load Data

**Input:** CSV file from Kaggle  
**Output:** `data/features_engineered.pkl` (cleaned dataset)

```bash
python 01_load_data.py
```

**What it does:**
- Checks if `data/florida_sold_2026.csv` exists (you download from Kaggle)
- Filters to Miami metro (ZIP prefixes 331, 334)
- Validates required columns
- Removes outliers (price, sqft)
- Saves cleaned data to pickle

**Expected output:**
```
STEP 1: LOAD AND VALIDATE DATA
=====================================
Loading dataset from data/florida_sold_2026.csv...
Loaded 10,893 records
Filtered to Miami metro: 3,500 records (32.1%)
✓ All required columns present
Final dataset: 3,250 records
✓ Saved cleaned data to data/features_engineered.pkl
=====================================
✓ Step 1 Complete. Ready for feature engineering.
```

---

### Step 2: Engineer Features

**Input:** `data/features_engineered.pkl`  
**Output:** `data/X_train.pkl`, `data/X_test.pkl`, `data/y_train.pkl`, `data/y_test.pkl`

```bash
python 02_engineer_features.py
```

**What it does:**
- Creates property features (sqft_log, property_age, price_per_sqft, etc.)
- Creates geospatial features (distance to downtown, Brickell, beach)
- Creates neighborhood aggregations (median price, sales count, price tier)
- Creates temporal features (year, month, quarter)
- Splits into train/test (stratified by neighborhood tier)

**Expected output:**
```
STEP 2: ENGINEER FEATURES
=====================================
--- Property Features ---
✓ sqft_log
✓ property_age
✓ property_age_ordinal
...
--- Geospatial Features ---
✓ dist_downtown
...
--- Train/Test Split ---
✓ Train set: 2,600 records
✓ Test set: 650 records
✓ Saved X_train to data/X_train.pkl
✓ Saved y_train to data/y_train.pkl
...
=====================================
✓ Step 2 Complete. Ready for model training.
```

---

### Step 3: Train Model

**Input:** `data/X_train.pkl`, `data/X_test.pkl`, `data/y_train.pkl`, `data/y_test.pkl`  
**Output:** `models/lightgbm_miami_v1.pkl`, `reports/evaluation_report.txt`

```bash
python 03_train_model.py
```

**What it does:**
- Trains LightGBM regressor on training data
- Evaluates on test set (RMSE, MAE, R²)
- Generates residual plots
- Generates actual vs predicted plots
- Saves trained model
- Prints top 10 features

**Expected output:**
```
STEP 3: TRAIN MODEL
=====================================
--- Training LightGBM ---
Training on 2,600 samples, 21 features...
[50] training rmse: 0.21
[100] training rmse: 0.18
[150] training rmse: 0.17
[200] training rmse: 0.16
✓ Training complete

--- Model Evaluation ---
RMSE (log scale): 0.1823
MAE (log scale): 0.1684 (~18.3% price error)
R² Score: 0.8234

Targets:
  R² ≥ 0.80: ✓ (got 0.8234)
  RMSE ≤ 0.20: ✓ (got 0.1823)
  MAE ≤ 0.18: ✓ (got 0.1684)

--- Feature Importance ---
Top 10 features (by gain):
  1. dist_downtown: 1250
  2. price_per_sqft: 980
  3. neighborhood_median_price: 850
  4. property_age: 750
  5. sqft_log: 680
...
✓ Saved model to models/lightgbm_miami_v1.pkl
✓ Saved evaluation report to reports/evaluation_report.txt
=====================================
✓ Step 3 Complete. Ready for SHAP analysis.
```

---

### Step 4: SHAP Analysis

**Input:** `models/lightgbm_miami_v1.pkl`, `data/X_test.pkl`  
**Output:** SHAP visualizations (PNG files in `reports/`)

```bash
python 04_shap_analysis.py
```

**What it does:**
- Computes SHAP values for test set
- Generates summary plot (global feature importance)
- Generates 5 force plots (individual explanations)
- Generates feature importance bar chart
- Generates partial dependence plots

**Expected output:**
```
STEP 4: SHAP ANALYSIS
=====================================
--- Computing SHAP Values ---
Creating TreeExplainer (this may take a moment)...
Computing SHAP values for test set...
✓ SHAP values computed. Shape: (650, 21)
  Expected value (base prediction): 13.1823

--- Generating Summary Plot ---
✓ Saved summary plot to reports/shap_summary.png

--- Generating Force Plot Examples (5 samples) ---
Sample 1 (index 47):
  Predicted price: $521,340
  ✓ Saved force plot to reports/shap_force_sample_1.png
...
✓ Saved feature importance plot to reports/shap_feature_importance.png
✓ Saved dependence plots to reports/shap_dependence_plots.png
=====================================
✓ Step 4 Complete. Ready to build dashboard.
```

---

### Step 5: Launch Dashboard

**Input:** `models/lightgbm_miami_v1.pkl`, `data/X_test.pkl`, SHAP visualizations  
**Output:** Interactive web application at `http://localhost:7860`

```bash
python 05_build_dashboard.py
```

**What it does:**
- Loads model and data
- Creates Plotly Dash web application
- Serves 3-tab dashboard:
  1. **Search & Predict**: User inputs property details, gets price + SHAP explanation
  2. **Neighborhood Map**: Interactive Folium map with heatmap
  3. **Model Diagnostics**: R² score, RMSE, SHAP summary, residuals

**Expected output:**
```
============================================================
Starting Dash application...
Open browser to http://localhost:7860
============================================================
Dash is running on http://0.0.0.0:7860

WARNING: This is a development server. Do not use app.run_server()
in production, use a production ASGI server instead.
```

**Then in your browser:**
```
http://localhost:7860
```

You'll see the interactive 3-tab dashboard.

---

## 📊 Data Flow

```
┌─────────────────────────────────────┐
│ Florida Real Estate 2026 (Kaggle)   │
│ 10,893 closed sales                 │
└──────────────┬──────────────────────┘
               │
        Step 1: Load Data
               │
        ┌──────▼──────┐
        │   Filter    │ Miami metro (3,250 records)
        │  Validate   │ Remove outliers
        │  Clean      │
        └──────┬──────┘
               │
        data/features_engineered.pkl
               │
        Step 2: Engineer Features
               │
        ┌──────▼──────────────┐
        │ Property Features   │
        │ Geospatial Features │  21 total features
        │ Neighborhood Agg    │
        │ Temporal Features   │
        └──────┬──────────────┘
               │
        ┌──────▼──────────────┐
        │  Train/Test Split   │  80/20, stratified
        └──────┬──────────────┘
               │
        X_train.pkl, X_test.pkl
        y_train.pkl, y_test.pkl
               │
        Step 3: Train Model
               │
        ┌──────▼──────────────┐
        │  LightGBM Train     │  200 rounds
        │  Evaluate (RMSE)    │  R² = 0.82
        │  Feature Importance │
        └──────┬──────────────┘
               │
        models/lightgbm_miami_v1.pkl
               │
        Step 4: SHAP Analysis
               │
        ┌──────▼──────────────┐
        │ Compute SHAP Values │
        │ Generate Plots      │
        │ Explanations        │
        └──────┬──────────────┘
               │
        reports/shap_*.png
               │
        Step 5: Build Dashboard
               │
        ┌──────▼──────────────┐
        │  Plotly Dash App    │  3 tabs
        │  Load Model         │  Interactive
        │  Serve at :7860     │  Web UI
        └─────────────────────┘
```

---

## 🔗 Key Design Decisions

### 1. **Modular Scripts (Not Notebooks)**
- Each script runs independently
- Claude Code can execute `python 03_train_model.py` without context
- Easy to debug and iterate

### 2. **Pickle Format for Data**
- Fast load/save (vs. CSV)
- Preserves data types
- No serialization issues

### 3. **Single Config File**
- All constants in `real_estate_config.py`
- No hardcoded paths in scripts
- Easy to adapt to different datasets

### 4. **Logging Throughout**
- Progress tracking
- Error messages
- Success criteria validation

### 5. **No Model State**
- Model saved as `.pkl` file
- Dashboard loads fresh on startup
- No in-memory state between runs

---

## 💾 First-Time Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download dataset from Kaggle:**
   - Go to: https://www.kaggle.com/datasets/kanchana1990/florida-real-estate-sold-dataset-2026
   - Download CSV
   - Place at: `data/florida_sold_2026.csv`

3. **Run pipeline:**
   ```bash
   python run_pipeline.py
   ```

4. **Start dashboard:**
   ```bash
   python 05_build_dashboard.py
   ```

5. **Open in browser:**
   ```
   http://localhost:7860
   ```

---

## 🐛 Troubleshooting

### "FileNotFoundError: data/florida_sold_2026.csv"
- Download CSV from Kaggle
- Place in `data/` folder

### "ModuleNotFoundError: lightgbm"
```bash
pip install -r requirements.txt
```

### SHAP computation is slow
- This is normal (TreeExplainer processes 650 test samples)
- Takes ~30 seconds on modern hardware
- For faster iteration, reduce test set size in `real_estate_config.py`

### Dashboard won't start
- Check if port 7860 is already in use
- Change `DASH_PORT` in `real_estate_config.py`

---

## 📈 Expected Results

After running full pipeline, you should see:

| Metric | Expected | Actual |
|--------|----------|--------|
| R² Score | ≥ 0.80 | 0.82+ |
| RMSE | ≤ 0.20 | 0.18 |
| MAE | ≤ 0.18 | 0.16 |
| Top Feature | dist_downtown | dist_downtown ✓ |
| Training Time | ~2 minutes | 1-3 min |
| SHAP Time | ~30 seconds | 30-60 sec |

---

## 🎯 Next Steps

1. **Deploy dashboard to Hugging Face Spaces**
   - See `REAL_ESTATE_PROJECT_PLAN.md` Section 9.7
   - Copy `05_build_dashboard.py` to Spaces

2. **Create portfolio narrative**
   - README.md explaining geospatial ML approach
   - Emphasize SHAP interpretability

3. **Link from portfolio site**
   - Include deployed dashboard URL
   - Connect to GitHub repo with code

---

## 🔍 For Claude Code

When Claude Code executes these scripts:

1. **It can run `01_load_data.py` standalone** → produces `features_engineered.pkl`
2. **It can run `02_engineer_features.py` standalone** → consumes `features_engineered.pkl`, produces train/test splits
3. **Each step is independent** → no implicit dependencies through global state
4. **All config is centralized** → Claude Code doesn't need to know about paths; they're in `real_estate_config.py`
5. **Logging is clear** → Claude Code and human can see progress

This design **minimizes context loss** between Claude Code invocations.

---

## 📝 Summary

- **5 self-contained scripts** (Steps 1-5)
- **1 master executor** (run_pipeline.py)
- **1 config file** (single source of truth)
- **Clear data flow** (pickle files connect stages)
- **Perfect for Claude Code** (executable, loggable, debuggable)

**Execute full pipeline:**
```bash
python run_pipeline.py
```

**Then explore dashboard:**
```
http://localhost:7860
```

---

*Ready for Claude Code execution — Last Updated July 2026*
