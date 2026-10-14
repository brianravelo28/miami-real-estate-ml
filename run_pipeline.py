"""
Pipeline executor: builds data/sales_clean.csv from the raw Kaggle file.
Run with: python run_pipeline.py   (needs pgeocode; see requirements-dev.txt)
"""

import subprocess
import sys

result = subprocess.run([sys.executable, 'src/01_load_data.py'])
sys.exit(result.returncode)
