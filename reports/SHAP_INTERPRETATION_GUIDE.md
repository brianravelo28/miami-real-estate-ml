# SHAP Interpretation Guide

## What is SHAP?
SHAP (SHapley Additive exPlanations) values explain a model's prediction by showing how much each feature pushes it above or below the base value (the model's average prediction).

Because the model predicts **log price**, SHAP values are in log-price units. A SHAP value of +0.10 means roughly +10% on the predicted price.

## Reading the dashboard's SHAP chart
- **Bars** are individual features' contributions to this prediction (top 10 by magnitude).
- **Positive** bars increase the predicted price; **negative** bars decrease it.
- The base value is about 13.20 in log space (roughly $540,973); the contributions sum to the final log prediction.

## Top Features (mean absolute SHAP value, test set)

| Rank | Feature | Mean abs SHAP |
|------|---------|---------------|
| 1 | `sqft` | 0.347 |
| 2 | `is_condo` | 0.095 |
| 3 | `lon` | 0.089 |
| 4 | `dist_downtown` | 0.066 |
| 5 | `sqft_log` | 0.061 |
| 6 | `lat` | 0.055 |
| 7 | `property_age` | 0.054 |
| 8 | `neighborhood_price_std` | 0.054 |

## Limitations
- SHAP assumes features can be varied independently, which doesn't hold for correlated features like `sqft`/`sqft_log` or `lat`/`lon`.
- Explanations are local to each prediction; the Diagnostics tab shows global importance.
- The model is trained on Miami-Dade and Broward 2026 sales and may not generalize elsewhere.
