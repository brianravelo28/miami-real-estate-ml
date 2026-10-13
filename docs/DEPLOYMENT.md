# Deployment Guide

The dashboard is a Streamlit app (`app.py`) deployed on Render's free tier:
**https://miami-real-estate-ml.onrender.com/**

---

## What the app needs at runtime

| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard |
| `models/lightgbm_miami_v1.pkl` | Trained model |
| `data/X_test.pkl`, `data/y_test.pkl` | Test split (metrics, SHAP, map) |
| `data/feature_artifacts.pkl` | Per-ZIP stats and ZIP centroids used to build prediction features |
| `requirements.txt` | Dependencies |

The pickles are excluded by `.gitignore` but committed anyway (`git add -f`, tracked via Git LFS per `.gitattributes`) so the deployed service has them. Re-run `python run_pipeline.py` after changing data or features, then force-add the regenerated pickles.

---

## Render (current deployment)

**Pros:** free tier, auto-deploys on every push to `main`, no card required.
**Cons:** free services sleep after ~15 minutes idle (first load afterward takes ~30 seconds) and have 512 MB RAM.

### Setup

1. Sign in at https://render.com with GitHub.
2. **New +** → **Web Service** → connect the `miami-real-estate-ml` repo.
3. Settings:
   - **Environment**: Python 3
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
   - **Plan**: Free
4. Deploy. Every push to `main` triggers a redeploy.

`render.yaml` in the repo mirrors these settings.

---

## Docker (optional)

```bash
docker build -t miami-real-estate-ml .
docker run -p 8501:8501 miami-real-estate-ml
# open http://localhost:8501
```

The `Dockerfile` copies only the files listed above, so it works for any container host (Cloud Run, Fly.io, ECS, etc.).

---

## Other options

| Platform | Cost | Notes |
|----------|------|-------|
| **Streamlit Community Cloud** | Free | Simplest for Streamlit apps; connects directly to the GitHub repo |
| **Hugging Face Spaces** | Free | Works with the Docker SDK, but free CPU quota is limited |
| **Cloud Run / Fly.io** | Pay-per-use | Use the Dockerfile |

---

## Troubleshooting

### "No such file or directory: models/lightgbm_miami_v1.pkl"
The pickles weren't pushed. Run `git add -f models/lightgbm_miami_v1.pkl data/X_test.pkl data/y_test.pkl data/feature_artifacts.pkl`, commit, and push. `app.py` resolves paths relative to its own location, so the working directory doesn't matter.

### Python and library versions
`.python-version` pins the Python version Render uses. `requirements.txt` gives every package a minimum version **and a cap at the next major version** (for example `pandas>=2.0.0,<4`), so a surprise major release can't silently break the pickled model or the dashboard on the next deploy, while minor and patch updates still flow in. A test (`tests/test_requirements.py`) enforces that every package keeps both bounds.

**Known-good versions** (what Render installed and ran successfully on 2026-10-13, read from the build log and the dashboard's Environment panel):

| Package | Render ran | Model was built with (local) |
|---------|-----------|------------------------------|
| Python | 3.14.3 | 3.9.13 |
| pandas | 3.0.6 | 2.0.3 |
| numpy | 2.5.4 | 1.24.3 |
| scikit-learn | 1.9.1 | 1.3.0 |
| LightGBM | 4.7.0 | 4.6.0 |
| SHAP | 0.53.0 | 0.49.1 |
| Streamlit | 1.65.0 | 1.50.0 |

The model gives identical predictions on both sets (checked with the default input: $607,572, range $443,187–$1,005,645), even across pandas 2→3 and numpy 1→2. If a future deploy misbehaves, the table above is the set to fall back to.

The dashboard's sidebar **Environment** panel shows the deployed commit (from Render's `RENDER_GIT_COMMIT`) and, for each library, the version the model was built with → the version running now, flagging only major-version differences. `models/model_metadata.json` (written by `src/03_train_model.py`) holds the build-time versions; commit it together with a regenerated model.

**Stale deploys:** once a deploy sat for 10 minutes and then reported live while the old version kept serving. **Manual Deploy → Clear build cache & deploy latest commit** fixed it (the redeploy took about 90 seconds). The deployed-commit line in the Environment panel makes this easy to spot.

### Pickle / pandas version errors
Pickles must be loaded with a compatible pandas version. If you see `ModuleNotFoundError: pandas.core.indexes.numeric`, regenerate the artifacts with the same Python/pandas versions the server uses (`python run_pipeline.py`) and push them again.

### Build fails on `requirements.txt`
Make sure the file is UTF-8. Appending to it with PowerShell's `>>` writes UTF-16 and breaks `pip`; edit it in an editor instead.

### Slow first load
The app computes SHAP values for the test set at startup (cached afterward with `st.cache_resource`), and the free tier cold-starts after idle periods.

### Out of memory
The free tier has 512 MB. The model and test data are small, but SHAP plus Streamlit can approach that limit; upgrade the plan if the service restarts under load.

---

## References

- [Streamlit deployment docs](https://docs.streamlit.io/deploy)
- [Render docs](https://render.com/docs)
