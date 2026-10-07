FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The dashboard only needs the app, trained model, and test split
COPY app.py .
COPY .streamlit/ .streamlit/
COPY data/X_test.pkl data/
COPY data/y_test.pkl data/
COPY data/feature_artifacts.pkl data/
COPY models/lightgbm_miami_v1.pkl models/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
