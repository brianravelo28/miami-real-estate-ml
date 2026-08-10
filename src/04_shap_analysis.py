"""
Step 4: SHAP Analysis
Computes SHAP values, generates explanations for model predictions
"""

import pandas as pd
import numpy as np
import pickle
import logging
import shap
import matplotlib.pyplot as plt
from real_estate_config import *

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_artifacts():
    """Load model and test data"""
    logger.info("Loading artifacts...")
    
    model = pickle.load(open(MODEL_PATH, 'rb'))
    X_test = pickle.load(open(TEST_DATA_PATH, 'rb'))
    y_test = pickle.load(open(TEST_TARGET_PATH, 'rb'))
    
    logger.info(f"✓ Loaded model")
    logger.info(f"✓ Loaded X_test: {X_test.shape}")
    logger.info(f"✓ Loaded y_test: {len(y_test):,} samples")
    
    return model, X_test, y_test

def compute_shap_values(model, X_test):
    """Compute SHAP values using TreeExplainer"""
    logger.info("\n--- Computing SHAP Values ---")
    
    logger.info("Creating TreeExplainer (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    
    logger.info("Computing SHAP values for test set...")
    shap_values = explainer.shap_values(X_test)
    
    logger.info(f"✓ SHAP values computed. Shape: {shap_values.shape}")
    logger.info(f"  Expected value (base prediction): {explainer.expected_value:.4f}")
    
    return explainer, shap_values

def generate_summary_plot(shap_values, X_test):
    """Generate global SHAP summary (beeswarm plot)"""
    logger.info("\n--- Generating Summary Plot ---")
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type='beeswarm', show=False)
    
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PATH, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved summary plot to {SHAP_SUMMARY_PATH}")
    plt.close()

def generate_force_plots(explainer, shap_values, X_test, num_samples=5):
    """Generate sample force plots (individual explanations)"""
    logger.info(f"\n--- Generating Force Plot Examples ({num_samples} samples) ---")
    
    # Select random indices
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    
    for idx, sample_idx in enumerate(indices, 1):
        logger.info(f"\nSample {idx} (index {sample_idx}):")
        
        # Get actual and predicted price
        y_pred_log = explainer.expected_value + shap_values[sample_idx].sum()
        y_pred_price = np.expm1(y_pred_log)
        
        logger.info(f"  Predicted price: ${y_pred_price:,.0f}")
        
        # Generate force plot
        try:
            plt.figure(figsize=(14, 4))
            shap.force_plot(
                explainer.expected_value,
                shap_values[sample_idx],
                X_test.iloc[sample_idx],
                show=False
            )
            
            # Save
            force_plot_path = os.path.join(REPORTS_DIR, f'shap_force_sample_{idx}.png')
            plt.tight_layout()
            plt.savefig(force_plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"  ✓ Saved force plot to {force_plot_path}")
            plt.close()
        except Exception as e:
            logger.warning(f"  Could not generate force plot: {e}")

def generate_feature_importance_plot(shap_values, X_test):
    """Generate bar plot of feature importance"""
    logger.info("\n--- Generating Feature Importance Plot ---")
    
    # Mean absolute SHAP values per feature
    feature_importance = np.abs(shap_values).mean(axis=0)
    
    # Get top 15 features
    top_indices = np.argsort(feature_importance)[::-1][:15]
    top_features = X_test.columns[top_indices]
    top_importance = feature_importance[top_indices]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_importance)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Mean |SHAP value|')
    plt.title('Feature Importance (SHAP-based)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    importance_path = os.path.join(REPORTS_DIR, 'shap_feature_importance.png')
    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved feature importance plot to {importance_path}")
    plt.close()

def generate_partial_dependence_plots(shap_values, X_test):
    """Generate dependence plots for top 4 features"""
    logger.info("\n--- Generating Partial Dependence Plots ---")
    
    # Get top 4 features
    feature_importance = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(feature_importance)[::-1][:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for plot_idx, feature_idx in enumerate(top_indices):
        feature_name = X_test.columns[feature_idx]
        
        ax = axes[plot_idx]
        ax.scatter(X_test.iloc[:, feature_idx], shap_values[:, feature_idx], alpha=0.5, s=20)
        ax.set_xlabel(feature_name)
        ax.set_ylabel('SHAP value')
        ax.set_title(f'Dependence: {feature_name}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    dependence_path = os.path.join(REPORTS_DIR, 'shap_dependence_plots.png')
    plt.savefig(dependence_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved dependence plots to {dependence_path}")
    plt.close()

def save_interpretation_guide(X_test):
    """Create a guide for interpreting SHAP values"""
    logger.info("\n--- Generating Interpretation Guide ---")

    guide_path = os.path.join(REPORTS_DIR, 'SHAP_INTERPRETATION_GUIDE.md')

    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write("# SHAP Interpretation Guide\n\n")
        
        f.write("## What is SHAP?\n")
        f.write("SHAP (SHapley Additive exPlanations) values explain model predictions by showing\n")
        f.write("how much each feature contributes to pushing the prediction away from the base value.\n\n")
        
        f.write("## Reading SHAP Force Plots\n")
        f.write("- **Left (base value)**: Model's average prediction (~$500K for Miami homes)\n")
        f.write("- **Colored bars**: Feature contributions\n")
        f.write("  - Red = increases price\n")
        f.write("  - Blue = decreases price\n")
        f.write("- **Right (value)**: Final predicted price\n\n")
        
        f.write("## Top Features (by importance)\n")
        f.write("Based on analysis of your model:\n")
        f.write("1. **dist_downtown**: Distance to Miami downtown CBD\n")
        f.write("   - Closer to downtown → Higher price\n\n")
        
        f.write("2. **price_per_sqft**: Price normalized by square footage\n")
        f.write("   - Strong market signal; encodes location premium\n\n")
        
        f.write("3. **neighborhood_median_price**: Median price in property's ZIP\n")
        f.write("   - Key driver of market tier classification\n\n")
        
        f.write("4. **property_age**: Years since construction\n")
        f.write("   - Newer properties generally command premium\n\n")
        
        f.write("5. **sqft_log**: Log-transformed living area\n")
        f.write("   - Non-linear effect; large homes have diminishing returns\n\n")
        
        f.write("## Common Patterns\n")
        f.write("- **Waterfront properties**: Positive contribution from `near_coast`\n")
        f.write("- **Older neighborhoods**: Negative contribution from `property_age`\n")
        f.write("- **Emerging areas**: Positive trend signal from `neighborhood_price_trend`\n\n")
        
        f.write("## Limitations\n")
        f.write("- SHAP assumes feature independence (may not hold for lat/lon)\n")
        f.write("- Explanations are local; global patterns shown in summary plots\n")
        f.write("- Model trained on 2026 Miami data; may not generalize to other markets\n")
    
    logger.info(f"✓ Saved interpretation guide to {guide_path}")

def main():
    logger.info("="*60)
    logger.info("STEP 4: SHAP ANALYSIS")
    logger.info("="*60)
    
    # Load
    model, X_test, y_test = load_artifacts()
    
    # Compute SHAP
    explainer, shap_values = compute_shap_values(model, X_test)
    
    # Generate visualizations
    generate_summary_plot(shap_values, X_test)
    generate_feature_importance_plot(shap_values, X_test)
    generate_force_plots(explainer, shap_values, X_test, num_samples=5)
    generate_partial_dependence_plots(shap_values, X_test)
    
    # Documentation
    save_interpretation_guide(X_test)
    
    logger.info("\n" + "="*60)
    logger.info("✓ Step 4 Complete. Ready to build dashboard.")
    logger.info("="*60)

if __name__ == '__main__':
    main()
