# Data Acquisition Guide

## Downloading the dataset

The project uses the **Florida Real Estate Sold 2026** dataset from Kaggle.

1. Go to https://www.kaggle.com/datasets/kanchana1990/florida-real-estate-sold-dataset-2026 (free Kaggle account required).
2. Download the CSV (`florida_real_estate_sold_properties_ultimate.csv`, ≈14 MB).
3. Place it at `data/florida_sold_2026.csv` (the scripts expect this name):

   ```bash
   mv ~/Downloads/florida_real_estate_sold_properties_ultimate.csv data/florida_sold_2026.csv
   ```

### Alternative: Kaggle CLI

```bash
kaggle datasets download -d kanchana1990/florida-real-estate-sold-dataset-2026
unzip florida-real-estate-sold-dataset-2026.zip -d data/
mv data/florida_real_estate_sold_properties_ultimate.csv data/florida_sold_2026.csv
```

(Set up credentials first by saving your API token to `~/.kaggle/kaggle.json`.)

---

## Dataset overview

| Aspect | Details |
|--------|---------|
| **Name** | Florida Real Estate Sold 2026 |
| **Source** | Kaggle |
| **Records** | 10,893 (statewide Florida) |
| **Records after filtering** | 990 (Miami-Dade + Broward, outliers and missing values removed) |
| **Columns** | 14 |
| **Format** | CSV |

Column definitions are in [`../docs/DATA_SCHEMA.md`](../docs/DATA_SCHEMA.md).

---

## Data limitations

1. **Geographic**: statewide data, filtered to Miami-Dade and Broward by the pipeline
2. **Temporal**: 2026 snapshot with no sale dates
3. **Coordinates**: none in the file; the pipeline geocodes each ZIP code to its centroid with `pgeocode` (needs internet access the first time it runs to download ZIP data)
4. **Flood zones**: no FEMA data; a latitude proxy is used
5. **Text descriptions**: included but unused

---

## How the pipeline uses this data

1. **Step 1 (`src/01_load_data.py`)**: reads the CSV, normalizes ZIPs, geocodes them, filters to Miami-Dade and Broward, validates and cleans, and writes `data/features_engineered.pkl` (990 records).
2. **Steps 2–4**: work from the pickled data; the raw CSV isn't needed after step 1.
3. **Dashboard (`app.py`)**: needs only the trained model and the test-split pickles.

---

## Storage

`data/florida_sold_2026.csv` is excluded by `.gitignore` (large, third-party data, easily re-downloaded). The pipeline is reproducible from the CSV, the code in `src/`, and `src/real_estate_config.py`.

---

## Quality checks

After downloading, the CSV should have 10,893 data rows and these 14 columns: `type`, `sub_type`, `listPrice`, `lastSoldPrice`, `sqft`, `stories`, `beds`, `baths`, `baths_full`, `baths_full_calc`, `garage`, `year_built`, `zip`, `sanitized_text`.

```bash
head -1 data/florida_sold_2026.csv      # column names
wc -l data/florida_sold_2026.csv        # line count (can exceed 10,894 if descriptions contain newlines)
```

---

## Next steps

```bash
python run_pipeline.py          # full pipeline

# or step by step
python src/01_load_data.py
python src/02_engineer_features.py
python src/03_train_model.py
python src/04_shap_analysis.py

streamlit run app.py            # launch the dashboard
```
