# Data Schema

## `data/sales_clean.csv` (990 rows, written by `src/01_load_data.py`)

| Column | Type | Description |
|--------|------|-------------|
| `type` | text | Raw property type: `single_family`, `condos`, `townhomes`, `condo_townhome_rowhome_coop`, `multi_family`, `mobile` |
| `sub_type` | text | Raw sub-type; about half empty and not used by the dashboard |
| `listPrice` | number | Last list price (USD) |
| `lastSoldPrice` | number | Sale price (USD), $50K–$10M after cleaning |
| `sqft` | number | Living area, 400–10,000 after cleaning |
| `beds`, `baths` | number | Bedrooms and bathrooms |
| `yr_built` | number | Year built (renamed from `year_built`; a few presales are dated 2027) |
| `zip` | text | 5-digit ZIP code |
| `county_name` | text | `Miami-Dade` or `Broward`, from the ZIP database |
| `lat`, `lon` | number | Center of the ZIP code (not the property's location) |

Source: Kaggle "Florida Real Estate Sold 2026". The raw file has 14 columns and 10,893 statewide rows; the listing description text is not kept.

## Columns added by `src/summaries.py` (in memory only)

| Column | Definition |
|--------|-----------|
| `property_type` | `type` grouped into Single family, Condo, Townhome, Other |
| `price_per_sqft` | `lastSoldPrice / sqft` |
| `build_period` | `yr_built` banded: before 1960, 1960–1979, 1980–1999, 2000–2014, 2015 or newer |

## Per-ZIP summary (`summaries.zip_summary`)

`sales` (count), `median_price`, `median_price_per_sqft`, `median_sqft`, ZIP center `lat`/`lon`, `county`, and `thin` (true when `sales` < `MIN_ZIP_SALES`, currently 5).
