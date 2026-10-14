# 🏠 Miami Real Estate Market Explorer

An interactive dashboard of ~1,000 recent home sales in Miami-Dade and Broward counties: price by ZIP code, by property type, and against size. It describes past sales; it does not predict prices (see [why](#why-theres-no-price-predictor)).

**Live Demo**: [miami-real-estate-ml.onrender.com](https://miami-real-estate-ml.onrender.com/) (free tier, so the first load may take a moment to wake up)

**Run locally**: `pip install -r requirements.txt`, then `streamlit run app.py` (opens at `http://localhost:8501`)

---

## What's in the dashboard

Sidebar filters (county, property type, price range) apply to every tab.

| Tab | Shows |
|-----|-------|
| **Overview** | Sales count, median price, median $/sqft, price distribution by property type, county and type summaries |
| **Map by ZIP** | One circle per ZIP: size = number of sales, color = median price (by rank). ZIPs with fewer than 5 sales are hidden by default and shaded grey if shown |
| **What goes with price** | Price vs size scatter (log scales) and median $/sqft by bedrooms, build period, county or type. Every bar shows its sample size; groups under 10 sales are hidden |
| **ZIP table** | Sales count, median price, median $/sqft and median size for every ZIP |
| **About the data** | Source, cleaning rules, limitations, and why there is no predictor |

## The data

- **Source:** [Florida Real Estate Sold 2026](https://www.kaggle.com/datasets/kanchana1990/florida-real-estate-sold-dataset-2026) (Kaggle), 10,893 statewide closed sales.
- **Miami-Dade + Broward only:** each ZIP is looked up in `pgeocode` and kept if its county is a target county (not by ZIP prefix; prefixes like 334 also cover Palm Beach and Monroe). A bounding-box check drops ZIPs the database places far from their county.
- **Cleaning:** sales of $50K–$10M and 400–10,000 sqft, with all key fields present.
- **Result:** **990 sales** (503 Miami-Dade, 487 Broward) in **128 ZIP codes**. The median ZIP has 7 sales and 91 ZIPs have at least 5.

Limitations: the file has no street addresses (the map places each ZIP at its center), no sale dates (one snapshot, no trends), and no HOA, tax, condition, view or lot-size fields. It is not a random sample of all sales. Medians for ZIPs with only a few sales mostly reflect which houses happened to sell.

## Why there's no price predictor

An earlier version of this project predicted prices with LightGBM and SHAP. It was removed on purpose:

1. **The first score was wrong.** It reported R² 0.99, but two features were computed from the sale price itself and ZIP price statistics included the test rows (target leakage). After removing them, the honest score was R² ≈ 0.82.
2. **0.82 is not precise.** The typical error was about 28%, and the 80% error band ran from roughly −29% to +63% of the estimate. Most of the signal came from square footage and property type; the model also overfit (train R² 0.97 vs test 0.82).
3. **Location was the weak part.** With ZIP-level coordinates and ~7 sales per ZIP, the same house's estimate moved by about 10% (up to ~30% in the worst tenth of cases) depending on which neighboring ZIP it was matched to. Shrinking ZIP statistics toward the county reduced that on average but not for every case, and how well a real address finds its true ZIP couldn't be validated because the data has no addresses.

Rather than keep adding caveats to a point estimate, the dashboard shows the sales themselves, with their sample sizes. The full predictor is in the git history (commit `13a8747`).

---

## Rebuild the data

```bash
pip install -r requirements-dev.txt                 # adds pytest and pgeocode
# download the Kaggle CSV to data/florida_sold_2026.csv (see data/acquisition.md)
python run_pipeline.py                              # writes data/sales_clean.csv
```

`data/sales_clean.csv` (the only file the dashboard reads) is committed, so the app runs without the raw download.

## Tests

```bash
python -m pytest
```

- `tests/test_data_prep.py`: the cleaning rules, with a fake geocoder so it runs offline: float ZIPs, county filtering instead of ZIP prefixes, the mislabeled-ZIP bounding box, price and size outliers.
- `tests/test_summaries.py`: ZIP counts and medians, the thin-ZIP flag, and a check that the committed dataset still matches what the dashboard expects.
- `tests/test_requirements.py`: every dependency has a minimum version and a next-major cap.

GitHub Actions runs the suite on every push.

## Project structure

```
├── app.py                  # Streamlit dashboard
├── run_pipeline.py         # Builds data/sales_clean.csv
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # + pytest, pgeocode
├── src/
│   ├── 01_load_data.py     # Load, geocode ZIPs, clean, save
│   ├── data_prep.py        # The cleaning rules (tested)
│   ├── summaries.py        # Medians and counts used by the dashboard (tested)
│   └── real_estate_config.py
├── data/
│   ├── acquisition.md      # How to download the Kaggle file
│   └── sales_clean.csv     # Cleaned sales (dashboard input)
├── tests/
└── docs/                   # METHODOLOGY, DATA_SCHEMA, ARCHITECTURE, DEPLOYMENT
```

## License

MIT, see [LICENSE](LICENSE).
