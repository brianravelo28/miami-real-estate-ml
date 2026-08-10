"""
Step 3: Train Model
Trains LightGBM on engineered features
Evaluates on test set, saves model artifacts
"""

import pandas as pd
import numpy as np
import pickle
import logging
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from real_estate_config import *

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load train/test data from step 2"""
    logger.info("Loading train/test data...")
    
    X_train = pickle.load(open(TRAIN_DATA_PATH, 'rb'))
    X_test = pickle.load(open(TEST_DATA_PATH, 'rb'))
    y_train = pickle.load(open(TRAIN_TARGET_PATH, 'rb'))
    y_test = pickle.load(open(TEST_TARGET_PATH, 'rb'))
    
    logger.info(f"✓ Loaded X_train: {X_train.shape}")
    logger.info(f"✓ Loaded X_test: {X_test.shape}")
    logger.info(f"✓ Loaded y_train: {len(y_train):,} samples")
    logger.info(f"✓ Loaded y_test: {len(y_test):,} samples")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train LightGBM regressor"""
    logger.info("\n--- Training LightGBM ---")
    
    # Create dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    
    logger.info(f"Training on {len(X_train):,} samples, {X_train.shape[1]} features...")
    
    # Train
    model = lgb.train(
        params=LIGHTGBM_PARAMS,
        train_set=train_data,
        num_boost_round=LIGHTGBM_NUM_ROUNDS,
        callbacks=[
            lgb.log_evaluation(period=50),
        ]
    )
    
    logger.info("✓ Training complete")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate on test set"""
    logger.info("\n--- Model Evaluation ---")
    
    # Predictions (log scale)
    y_pred = model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Convert to percentage error for interpretability
    # e^(MAE) - 1 ≈ percentage error in original price
    pct_error = (np.exp(mae) - 1) * 100
    
    logger.info(f"RMSE (log scale): {rmse:.4f}")
    logger.info(f"MAE (log scale): {mae:.4f} (~{pct_error:.1f}% price error)")
    logger.info(f"R² Score: {r2:.4f}")
    
    # Check against targets
    logger.info(f"\nTargets:")
    logger.info(f"  R² ≥ {TARGET_R2}: {'✓' if r2 >= TARGET_R2 else '✗'} (got {r2:.4f})")
    logger.info(f"  RMSE ≤ {TARGET_RMSE}: {'✓' if rmse <= TARGET_RMSE else '✗'} (got {rmse:.4f})")
    logger.info(f"  MAE ≤ {TARGET_MAE}: {'✓' if mae <= TARGET_MAE else '✗'} (got {mae:.4f})")
    
    return y_pred, rmse, mae, r2

def feature_importance(model, X_test):
    """Print top features"""
    logger.info("\n--- Feature Importance ---")
    
    # Get feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_names = X_test.columns
    
    # Sort
    indices = np.argsort(importance)[::-1][:10]  # Top 10
    
    logger.info("Top 10 features (by gain):")
    for rank, idx in enumerate(indices, 1):
        logger.info(f"  {rank}. {feature_names[idx]}: {importance[idx]:.0f}")
    
    return dict(zip(feature_names[indices], importance[indices]))

def plot_residuals(y_test, y_pred):
    """Generate residual plots"""
    logger.info("\n--- Generating Residual Plots ---")
    
    residuals = y_test.values - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted (log scale)')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted Values')
    axes[0].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[1].hist(residuals, bins=40, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Residuals')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(RESIDUAL_PLOT_PATH, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved residual plots to {RESIDUAL_PLOT_PATH}")
    plt.close()

def plot_actual_vs_predicted(y_test, y_pred):
    """Generate actual vs predicted plot"""
    logger.info("\n--- Generating Actual vs Predicted Plot ---")
    
    # Convert back from log scale for visualization
    y_test_price = np.expm1(y_test)
    y_pred_price = np.expm1(y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(y_test_price, y_pred_price, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_test_price.min(), y_pred_price.min())
    max_val = max(y_test_price.max(), y_pred_price.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    ax.set_xlabel('Actual Price ($)')
    ax.set_ylabel('Predicted Price ($)')
    ax.set_title('Actual vs Predicted House Prices')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format axes as currency
    ax.ticklabel_format(style='plain', axis='both')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved actual vs predicted plot to {os.path.join(REPORTS_DIR, 'actual_vs_predicted.png')}")
    plt.close()

def save_model(model):
    """Save trained model"""
    pickle.dump(model, open(MODEL_PATH, 'wb'))
    logger.info(f"\n✓ Saved model to {MODEL_PATH}")

def save_evaluation_report(rmse, mae, r2, top_features):
    """Save evaluation report as text"""
    
    report_path = os.path.join(REPORTS_DIR, 'evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("REAL ESTATE MODEL - EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("METRICS\n")
        f.write("-"*60 + "\n")
        f.write(f"RMSE (log scale): {rmse:.4f}\n")
        f.write(f"MAE (log scale): {mae:.4f}\n")
        f.write(f"R² Score: {r2:.4f}\n\n")
        
        f.write("TOP 10 FEATURES\n")
        f.write("-"*60 + "\n")
        for rank, (feature, importance) in enumerate(top_features.items(), 1):
            f.write(f"{rank}. {feature}: {importance:.0f}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    logger.info(f"✓ Saved evaluation report to {report_path}")

def main():
    logger.info("="*60)
    logger.info("STEP 3: TRAIN MODEL")
    logger.info("="*60)
    
    # Load
    X_train, X_test, y_train, y_test = load_data()
    
    # Train
    model = train_model(X_train, y_train)
    
    # Evaluate
    y_pred, rmse, mae, r2 = evaluate_model(model, X_test, y_test)
    
    # Feature importance
    top_features = feature_importance(model, X_test)
    
    # Diagnostics
    plot_residuals(y_test, y_pred)
    plot_actual_vs_predicted(y_test, y_pred)
    
    # Save
    save_model(model)
    save_evaluation_report(rmse, mae, r2, top_features)
    
    logger.info("\n" + "="*60)
    logger.info("✓ Step 3 Complete. Ready for SHAP analysis.")
    logger.info("="*60)

if __name__ == '__main__':
    main()
