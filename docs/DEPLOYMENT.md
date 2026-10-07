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
