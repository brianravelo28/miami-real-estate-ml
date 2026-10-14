# Deployment Guide

The dashboard is a Streamlit app (`app.py`) deployed on Render's free tier:
**https://miami-real-estate-ml.onrender.com/**

## What the app needs at runtime

| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard |
| `src/real_estate_config.py`, `src/summaries.py` | Paths and the summary tables |
| `data/sales_clean.csv` | The cleaned sales (about 85 KB, committed) |
| `requirements.txt` | Dependencies |

## Render

**Pros:** free tier, auto-deploys on every push to `main`, no card required.
**Cons:** free services sleep after ~15 minutes idle (the first load afterward takes ~30 seconds).

Settings (in the Render dashboard; the repo has no `render.yaml`):

- **Environment**: Python 3 (version pinned by `.python-version`)
- **Build command**: `pip install -r requirements.txt`
- **Start command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- **Plan**: Free

Other hosts such as Streamlit Community Cloud work the same way, since the app is a single script plus one CSV.

## Troubleshooting

**"Could not load sales_clean.csv"**: the file isn't in the deployed commit. Rebuild with `python run_pipeline.py` and commit `data/sales_clean.csv` (it is whitelisted in `.gitignore`).

**Library versions:** `requirements.txt` gives every package a minimum version and a cap at the next major version (e.g. `pandas>=2.0.0,<4`), so a surprise major release can't break a deploy while minor updates still flow in. `tests/test_requirements.py` enforces both bounds. Render has run Python 3.14 with pandas 3 and NumPy 2; the test suite also passes on Python 3.9 / pandas 2.

**Stale deploys:** once Render reported a new commit live while the old version kept serving. **Manual Deploy → Clear build cache & deploy latest commit** fixed it.

**Build fails on `requirements.txt`:** make sure the file is UTF-8. Appending with PowerShell's `>>` writes UTF-16 and breaks `pip`.

## References

- [Streamlit deployment docs](https://docs.streamlit.io/deploy)
- [Render docs](https://render.com/docs)
