# Deployment Guide

Deploy your interactive dashboard publicly so anyone can access it online!

---

## 🏆 Recommended: Hugging Face Spaces (FREE & EASIEST)

**Pros:**
- ✅ Completely free (no credit card needed)
- ✅ Zero configuration
- ✅ Automatic HTTPS
- ✅ Git-based deployment (push → auto-deploy)
- ✅ Built for ML demos
- ✅ Instant sharing (URL in seconds)

**Cons:**
- ⚠️ 48-hour auto-sleep on free tier (wakes up on access)
- ⚠️ 7GB storage limit

### Setup (10 minutes)

#### Step 1: Create Hugging Face Account
1. Go to https://huggingface.co/join
2. Sign up with email or GitHub

#### Step 2: Create a Space
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in:
   - **Space name**: `miami-real-estate-ml`
   - **License**: OpenRAIL (or your choice)
   - **Space SDK**: `Docker` (since we have requirements.txt)
   - **Visibility**: Public
4. Click "Create Space"

#### Step 3: Create Dockerfile
In your repo root, create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy files
COPY requirements.txt .
COPY src/ src/
COPY data/X_test.pkl data/
COPY data/y_test.pkl data/
COPY models/lightgbm_miami_v1.pkl models/
COPY reports/ reports/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 7860

# Run dashboard
CMD ["python", "src/05_build_dashboard.py"]
```

#### Step 4: Push to Hugging Face
```bash
cd "C:\Users\Brian\Desktop\Claude Projects\Real Estate Project"

# Add HF remote
git remote add huggingface https://huggingface.co/spaces/brianravelo28/miami-real-estate-ml

# Push to HF (same GitHub credentials)
git push huggingface main
```

**That's it!** 🎉 Your dashboard will be live at:
```
https://huggingface.co/spaces/brianravelo28/miami-real-estate-ml
```

---

## 🚀 Alternative: Render (FREE with auto-sleep)

**Pros:**
- ✅ Free tier with auto-deploy
- ✅ GitHub-integrated (auto-redeploy on push)
- ✅ Better uptime than HF Spaces (15-min auto-sleep)
- ✅ Custom domain support

**Cons:**
- ⚠️ Auto-sleep after 15 minutes of inactivity
- ⚠️ Cold start takes 30 seconds

### Setup (15 minutes)

#### Step 1: Push to GitHub (Already done ✓)

#### Step 2: Create Render Account
1. Go to https://render.com
2. Sign up with GitHub (easiest)

#### Step 3: Create Web Service
1. Click "New +" → "Web Service"
2. Connect your `miami-real-estate-ml` GitHub repo
3. Fill in:
   - **Name**: `miami-real-estate-ml`
   - **Environment**: Python 3
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `gunicorn --workers 1 --threads 4 --timeout 0 --bind 0.0.0.0:10000 "src.05_build_dashboard:app.server"`
   - **Plan**: Free

#### Step 4: Update Dashboard Port
In `src/05_build_dashboard.py`, change:
```python
DASH_PORT = int(os.environ.get('PORT', 10000))  # Render uses PORT env var
```

#### Step 5: Deploy
- Render auto-deploys on GitHub push
- Dashboard appears at: `https://miami-real-estate-ml.onrender.com`

---

## ☁️ Premium Options

### AWS (Elastic Beanstalk or EC2)
**Cost:** $3-10/month  
**Uptime:** 99.9%  
**Best for:** Production use

```bash
# Install AWS CLI
pip install awsebcli

# Initialize
eb init -p python-3.9 miami-real-estate-ml

# Create environment
eb create prod --instance-type t3.micro

# Deploy
eb deploy
```

### Google Cloud Run
**Cost:** Free tier + pay-per-use  
**Uptime:** 99.95%  
**Best for:** Serverless, pay-only-when-used

```bash
# Install gcloud CLI
# Then: gcloud app deploy
```

### Heroku (Paid)
**Cost:** $7-50/month (free tier discontinued)  
**Uptime:** 99.9%  
**Best for:** Rapid prototyping

---

## 📦 Before Deployment: Prepare Files

### 1. Create `Dockerfile` (for HF/Render/Cloud)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files
COPY requirements.txt .
COPY run_pipeline.py .
COPY src/ src/
COPY data/X_test.pkl data/
COPY data/y_test.pkl data/
COPY models/lightgbm_miami_v1.pkl models/
COPY reports/ reports/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (HF Spaces uses 7860, others use 10000)
EXPOSE 7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/ || exit 1

# Run dashboard
CMD ["python", "src/05_build_dashboard.py"]
```

### 2. Update Dashboard for Production

In `src/05_build_dashboard.py`, change the last line:

```python
if __name__ == '__main__':
    logger.info("="*60)
    logger.info("Starting Dash application...")
    logger.info(f"Open browser to http://localhost:{DASH_PORT}")
    logger.info("="*60)
    
    # Production: Use gunicorn or similar
    # Development: Direct Dash run
    app.run(
        host='0.0.0.0',  # Listen on all interfaces (required for deployment)
        port=int(os.environ.get('PORT', DASH_PORT)),  # Use env var if set
        debug=False  # Turn OFF debug in production!
    )
```

### 3. Update requirements.txt (add gunicorn for production)

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
shap>=0.40.0
matplotlib>=3.4.0
plotly>=5.0.0
dash>=2.0.0
geopy>=2.2.0
pillow>=8.3.0
gunicorn>=20.1.0  # Production ASGI server
```

---

## 🔍 Comparison Table

| Platform | Cost | Setup | Uptime | Cold Start | Auto-Deploy |
|----------|------|-------|--------|-----------|-------------|
| **Hugging Face Spaces** | Free | 10 min | 95% (sleep) | 5-10s | ✅ Git push |
| **Render** | Free | 15 min | 98% (sleep) | 10-30s | ✅ Git push |
| **AWS Elastic Beanstalk** | $3-10 | 30 min | 99.9% | 1-2s | ✅ Git push |
| **Google Cloud Run** | Pay-per-use | 20 min | 99.95% | 2-5s | ✅ Git push |
| **Heroku** | $7+ | 20 min | 99.9% | 1-2s | ✅ Git push |

---

## 🧪 Test Deployment Locally

Before deploying to production, test with Docker:

```bash
# Build Docker image
docker build -t miami-real-estate-ml .

# Run locally
docker run -p 7860:7860 miami-real-estate-ml

# Access at http://localhost:7860
```

---

## 🚨 Troubleshooting

### "Port already in use"
```python
# In src/05_build_dashboard.py, use environment variable:
port = int(os.environ.get('PORT', DASH_PORT))
app.run(port=port)
```

### "Model not found"
Ensure these files are copied in Dockerfile:
- `models/lightgbm_miami_v1.pkl`
- `data/X_test.pkl`, `data/y_test.pkl`
- `reports/*.png` (SHAP plots)

### "Out of memory"
Dashboard loads entire model + data into memory (~500MB).
- HF Spaces: 16GB available ✓
- Render free: 512MB (may be tight)
- AWS micro: 1GB (borderline)

### "Dashboard times out"
SHAP computation is slow (30-60 seconds).
- Consider pre-computing SHAP values offline
- Cache results to disk
- Use simpler visualizations on first load

---

## 📊 Post-Deployment Monitoring

### Set Up Monitoring
```bash
# Check uptime
# - New Relic (free tier)
# - Uptime Robot (free)
# - Healthchecks.io (free)

# Log errors
# - Sentry (free tier)
# - CloudWatch (AWS)
# - Stackdriver (GCP)
```

### Share Your Dashboard
Once deployed, you can share:
- **Direct link**: `https://huggingface.co/spaces/brianravelo28/miami-real-estate-ml`
- **Embed in portfolio**: Add `<iframe>` to portfolio website
- **Social media**: Share link with preview image
- **Resume**: "Live ML Dashboard: [link]"

---

## 🎯 Recommended Path

For **portfolio/demo** purposes:
1. ✅ **Start with Hugging Face Spaces** (simplest, free)
2. If you need better uptime → Render
3. If you need production reliability → AWS

---

## 📚 References

- [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Render Deployment Guide](https://render.com/docs)
- [AWS Elastic Beanstalk Python](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create-deploy-python-django.html)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Dash Production Deployment](https://dash.plotly.com/deployment)
