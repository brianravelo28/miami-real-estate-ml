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

def save_interpretation_guide(X_test, shap_values, base_value):
    """Write a guide for interpreting SHAP values, using this model's actual results"""
    logger.info("\n--- Generating Interpretation Guide ---")

    guide_path = os.path.join(REPORTS_DIR, 'SHAP_INTERPRETATION_GUIDE.md')
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X_test.columns)
    importance = importance.sort_values(ascending=False)
    base = float(np.ravel(base_value)[0])

    lines = [
        "# SHAP Interpretation Guide",
        "",
        "## What is SHAP?",
        "SHAP (SHapley Additive exPlanations) values explain a model's prediction by showing how much "
        "each feature pushes it above or below the base value (the model's average prediction).",
        "",
        "Because the model predicts **log price**, SHAP values are in log-price units. "
        "A SHAP value of +0.10 means roughly +10% on the predicted price.",
        "",
        "## Reading the dashboard's SHAP chart",
        "- **Bars** are individual features' contributions to this prediction (top 10 by magnitude).",
        "- **Positive** bars increase the predicted price; **negative** bars decrease it.",
        f"- The base value is about {base:.2f} in log space (roughly ${np.expm1(base):,.0f}); "
        "the contributions sum to the final log prediction.",
        "",
        "## Top Features (mean absolute SHAP value, test set)",
        "",
        "| Rank | Feature | Mean abs SHAP |",
        "|------|---------|---------------|",
    ]
    for rank, (name, value) in enumerate(importance.head(8).items(), 1):
        lines.append(f"| {rank} | `{name}` | {value:.3f} |")
    lines += [
        "",
        "## Limitations",
        "- SHAP assumes features can be varied independently, which doesn't hold for correlated features "
        "like `sqft`/`sqft_log` or `lat`/`lon`.",
        "- Explanations are local to each prediction; the Diagnostics tab shows global importance.",
        "- The model is trained on Miami-Dade and Broward 2026 sales and may not generalize elsewhere.",
        "",
    ]

    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

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
    save_interpretation_guide(X_test, shap_values, explainer.expected_value)
    
    logger.info("\n" + "="*60)
    logger.info("✓ Step 4 Complete. Run the dashboard with: streamlit run app.py")
    logger.info("="*60)

if __name__ == '__main__':
    main()
