# Data Acquisition Guide

## 📥 Downloading the Dataset

The project uses the **Florida Real Estate Sold 2026** dataset from Kaggle.

### Steps

1. **Visit Kaggle**:
   - Go to: https://www.kaggle.com/datasets/kanchana1990/florida-real-estate-sold-dataset-2026

2. **Download CSV**:
   - Click "Download" button
   - File: `florida_real_estate_sold_properties_ultimate.csv` (≈14 MB)
   - Requires Kaggle account (free signup)

3. **Place in Project**:
   ```bash
   # Move downloaded file to data folder
   mv ~/Downloads/florida_real_estate_sold_properties_ultimate.csv \
      /path/to/miami-real-estate-ml/data/
   
   # Rename for consistency (script expects this name)
   mv data/florida_real_estate_sold_properties_ultimate.csv \
      data/florida_sold_2026.csv
   ```

4. **Verify**:
   ```bash
   ls -lh data/florida_sold_2026.csv
   # Should show ~14 MB file
   ```

### Alternative: Kaggle CLI

If you have `kaggle` CLI installed:

```bash
# Authenticate (first time only)
kaggle auth login  # Creates ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d kanchana1990/florida-real-estate-sold-dataset-2026

# Unzip
unzip florida-real-estate-sold-dataset-2026.zip -d data/

# Rename
mv data/florida_real_estate_sold_properties_ultimate.csv data/florida_sold_2026.csv
```

---

## 📊 Dataset Overview

| Aspect | Details |
|--------|---------|
| **Name** | Florida Real Estate Sold 2026 |
| **Source** | Kaggle |
| **Size** | ≈14 MB |
| **Records** | 10,893 (statewide Florida) |
| **Records (Miami filtered)** | 1,068 (after filtering to ZIP 331, 334) |
| **Columns** | 14 |
| **Format** | CSV |

---

## 📋 Column Reference

See [`../docs/DATA_SCHEMA.md`](../docs/DATA_SCHEMA.md) for detailed column definitions.

---

## ⚠️ Data Limitations

1. **Geographic**: Florida statewide data; filtered to Miami metro manually
2. **Temporal**: 2026 snapshot; no historical trends
3. **Coordinates**: No lat/lon provided; project generates synthetic values from ZIP codes
4. **Flood zones**: No FEMA flood zone data; using latitude as proxy
5. **Images/Descriptions**: Includes text descriptions but no structured features

---

## 🔄 How the Pipeline Uses This Data

1. **Step 1 (Load)**: 
   - Reads CSV
   - Filters to Miami metro (ZIP prefix 331, 334)
   - Validates columns
   - Outputs: `features_engineered.pkl` (1,068 records)

2. **Steps 2–5**: 
   - Use pickled data (faster I/O)
   - Raw CSV not needed after Step 1

---

## 💾 Storage & Cleanup

### Don't Commit Raw Data to Git

The `.gitignore` file excludes:
```
data/*.csv          # Raw CSV files
data/raw/           # Raw data folder
```

**Why?**
- Files are large (14 MB)
- Copyrighted data
- Not needed for reproducibility (easily downloaded from Kaggle)

### To Regenerate:
```bash
# If you delete the CSV, re-download from Kaggle (steps above)
# The entire pipeline is reproducible from:
# 1. This CSV
# 2. The code in src/
# 3. The config in real_estate_config.py
```

---

## 🔍 Quality Checks

After downloading, the CSV should have:
- **10,893 rows** (including header)
- **14 columns**: type, sub_type, listPrice, lastSoldPrice, sqft, stories, beds, baths, baths_full, baths_full_calc, garage, year_built, zip, sanitized_text
- **No encoding issues** (UTF-8)

Verify:
```bash
# Check row count
wc -l data/florida_sold_2026.csv
# Should output: 10894 (10,893 rows + 1 header)

# Check columns
head -1 data/florida_sold_2026.csv
# Should show column names

# Check file size
du -h data/florida_sold_2026.csv
# Should be ≈14 MB
```

---

## 🚀 Next Steps

Once the CSV is downloaded and placed in `data/florida_sold_2026.csv`:

```bash
# Run the full pipeline
python run_pipeline.py

# Or run individual steps
python src/01_load_data.py
python src/02_engineer_features.py
python src/03_train_model.py
python src/04_shap_analysis.py
python src/05_build_dashboard.py
```

---

## 📞 Support

- **Download Issues**: See Kaggle account settings (API key, authentication)
- **File Not Found**: Double-check path and filename (`florida_sold_2026.csv`)
- **Encoding Errors**: Ensure CSV is UTF-8 (Windows may default to cp1252)

---

## 📚 References

- [Kaggle Dataset](https://www.kaggle.com/datasets/kanchana1990/florida-real-estate-sold-dataset-2026)
- [Kaggle CLI Docs](https://github.com/Kaggle/kaggle-api)
- [CSV Format Reference](https://tools.ietf.org/html/rfc4180)
