"""
Master Pipeline Executor
Runs all steps in sequence: Load → Features → Train → SHAP → Dashboard
Use this to execute the full pipeline with: python run_pipeline.py
"""

import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

STEPS = [
    {
        'name': 'Step 1: Load Data',
        'script': 'src/01_load_data.py',
        'description': 'Download and validate Florida real estate dataset'
    },
    {
        'name': 'Step 2: Engineer Features',
        'script': 'src/02_engineer_features.py',
        'description': 'Create geospatial and neighborhood features'
    },
    {
        'name': 'Step 3: Train Model',
        'script': 'src/03_train_model.py',
        'description': 'Train LightGBM regressor on engineered features'
    },
    {
        'name': 'Step 4: SHAP Analysis',
        'script': 'src/04_shap_analysis.py',
        'description': 'Generate SHAP explanations and visualizations'
    },
    {
        'name': 'Step 5: Build Dashboard',
        'script': 'src/05_build_dashboard.py',
        'description': 'Launch interactive Plotly Dash application'
    },
]

def run_step(step_num, script, name, description):
    """Execute a single step"""
    
    logger.info("\n" + "="*70)
    logger.info(f"{name}")
    logger.info("="*70)
    logger.info(f"Description: {description}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script],
            check=True,
            capture_output=False
        )
        logger.info(f"\n✓ {name} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"\n✗ {name} failed with exit code {e.returncode}\n")
        return False
    except FileNotFoundError:
        logger.error(f"\n✗ Could not find script: {script}\n")
        return False

def main():
    logger.info("\n" + "="*70)
    logger.info("REAL ESTATE PROJECT - FULL PIPELINE EXECUTION")
    logger.info("="*70)
    
    # Check if config exists
    if not os.path.exists('src/real_estate_config.py'):
        logger.error("\n✗ ERROR: src/real_estate_config.py not found!")
        logger.error("   Make sure you're running from the project root directory.\n")
        return False
    
    # Run steps
    completed = 0
    failed = 0
    
    for step_num, step in enumerate(STEPS, 1):
        success = run_step(
            step_num,
            step['script'],
            step['name'],
            step['description']
        )
        
        if success:
            completed += 1
        else:
            failed += 1
            # Optionally stop on first failure
            logger.warning(f"\nContinuing to next step despite failure...")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*70)
    logger.info(f"Completed: {completed}/{len(STEPS)}")
    logger.info(f"Failed: {failed}/{len(STEPS)}")
    
    if failed == 0:
        logger.info("\n✓ ALL STEPS COMPLETED SUCCESSFULLY!")
        logger.info("\nNext: Start the dashboard with:")
        logger.info("  python src/05_build_dashboard.py")
        logger.info("\nThen open browser to: http://localhost:7860")
    else:
        logger.error(f"\n✗ Pipeline failed. Please review errors above.")
    
    logger.info("="*70 + "\n")
    
    return failed == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
