# SHAP Interpretation Guide

## What is SHAP?
SHAP (SHapley Additive exPlanations) values explain model predictions by showing
how much each feature contributes to pushing the prediction away from the base value.

## Reading SHAP Force Plots
- **Left (base value)**: Model's average prediction (~$500K for Miami homes)
- **Colored bars**: Feature contributions
  - Red = increases price
  - Blue = decreases price
- **Right (value)**: Final predicted price

## Top Features (by importance)
Based on analysis of your model:
1. **dist_downtown**: Distance to Miami downtown CBD
   - Closer to downtown → Higher price

2. **price_per_sqft**: Price normalized by square footage
   - Strong market signal; encodes location premium

3. **neighborhood_median_price**: Median price in property's ZIP
   - Key driver of market tier classification

4. **property_age**: Years since construction
   - Newer properties generally command premium

5. **sqft_log**: Log-transformed living area
   - Non-linear effect; large homes have diminishing returns

## Common Patterns
- **Waterfront properties**: Positive contribution from `near_coast`
- **Older neighborhoods**: Negative contribution from `property_age`
- **Emerging areas**: Positive trend signal from `neighborhood_price_trend`

## Limitations
- SHAP assumes feature independence (may not hold for lat/lon)
- Explanations are local; global patterns shown in summary plots
- Model trained on 2026 Miami data; may not generalize to other markets
