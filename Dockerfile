FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
COPY src/ src/
COPY data/X_test.pkl data/
COPY data/y_test.pkl data/
COPY models/lightgbm_miami_v1.pkl models/
COPY reports/ reports/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (7860 for HF Spaces, can be overridden)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run dashboard
CMD ["python", "src/05_build_dashboard.py"]
