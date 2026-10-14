# Architecture

```
Kaggle CSV (data/florida_sold_2026.csv, not committed)
    │  src/01_load_data.py  (uses src/data_prep.py)
    │    normalize ZIPs → look up county + center point (pgeocode)
    │    → keep Miami-Dade + Broward → drop outliers / missing fields
    ▼
data/sales_clean.csv   (990 rows, committed)
    │  app.py  (uses src/summaries.py)
    ▼
Streamlit dashboard: Overview · Map by ZIP · What goes with price · ZIP table · About
```

| Piece | Role |
|-------|------|
| `src/real_estate_config.py` | Paths, price/size limits, target counties, the small-ZIP threshold |
| `src/data_prep.py` | Pure functions for each cleaning rule (the geocoder is injectable so tests run offline) |
| `src/01_load_data.py` | Script: read raw file, call `data_prep`, write `sales_clean.csv` |
| `src/summaries.py` | Derived columns and median/count tables used by the app |
| `app.py` | UI only; reads `data/sales_clean.csv`, applies the sidebar filters, draws tables, Plotly charts and the Folium map |
| `tests/` | Cleaning rules, summaries, dependency bounds |

Runtime needs only `app.py`, `src/real_estate_config.py`, `src/summaries.py`, `data/sales_clean.csv` and `requirements.txt`. `pgeocode` and the raw CSV are only needed to rebuild the data.
