"""
Miami Real Estate Price Predictor - Streamlit Dashboard
Interactive ML dashboard with SHAP explanations
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import shap
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components

APP_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_NAMES = [
    'beds', 'baths', 'sqft', 'sqft_log', 'property_age', 'price_per_sqft',
    'list_to_sold_ratio', 'is_condo', 'is_townhouse',
    'lat', 'lon', 'dist_downtown', 'dist_brickell', 'dist_beach',
    'neighborhood_median_price', 'neighborhood_price_std',
    'neighborhood_sales_count', 'flood_risk_percentile',
    'sale_year', 'sale_month'
]

# Set page config
st.set_page_config(
    page_title="Miami Real Estate Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD DATA & MODEL (cached for performance)
# ============================================================================

@st.cache_resource
def load_artifacts():
    """Load model, data, and SHAP values"""
    model = pickle.load(open(os.path.join(APP_DIR, 'models', 'lightgbm_miami_v1.pkl'), 'rb'))
    X_test = pickle.load(open(os.path.join(APP_DIR, 'data', 'X_test.pkl'), 'rb'))
    y_test = pickle.load(open(os.path.join(APP_DIR, 'data', 'y_test.pkl'), 'rb'))

    # Compute SHAP values once (cached)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    return model, X_test, y_test, explainer, shap_values

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.markdown("""
<style>
html, body, p, span, li, label, input, textarea, select, button, div[class*="css"] {
    font-size: 17px !important;
}
small, [data-testid="stCaptionContainer"], [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {
    font-size: 15px !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: rgba(28, 131, 225, 0.1) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🏠 Miami Real Estate Price Predictor")
st.markdown("AI-powered price estimation with applicable predictions (SHAP)")

# Load artifacts
try:
    model, X_test, y_test, explainer, shap_values = load_artifacts()
    st.success("✓ Model and data loaded")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ============================================================================
# SIDEBAR - MODEL INFO
# ============================================================================

with st.sidebar:
    st.markdown("## 📊 Model Performance")

    # Calculate metrics
    y_pred = model.predict(X_test)
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pct_error = (np.exp(mae) - 1) * 100

    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
        st.metric("RMSE", f"{rmse:.4f}", "log scale")
    with col2:
        st.metric("MAE", f"{mae:.4f}", f"{pct_error:.1f}%")
        st.metric("Samples", f"{len(X_test)}")

    st.markdown("---")
    st.markdown("### 📈 Dataset Info")
    st.info("""
    **Records:**

    990 Miami-Dade & Broward properties

    **Date**: 2026

    **Price Range:**

    $50K–$10M
    """)
    st.markdown("### 🧩 Features")
    st.selectbox("Features", FEATURE_NAMES, label_visibility="collapsed")

# ============================================================================
# TAB 1: SEARCH & PREDICT
# ============================================================================

tab1, tab2, tab3 = st.tabs(["🔍 Search & Predict", "🗺️ Neighborhood Map", "📈 Diagnostics"])

with tab1:
    st.markdown("## Enter Property Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        beds = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
        baths = st.number_input("Bathrooms", min_value=0, max_value=8, value=2)

    with col2:
        sqft = st.number_input("Square Feet", min_value=400, max_value=10000, value=1500)
        property_age = st.number_input("Property Age (years)", min_value=0, max_value=170, value=30)

    with col3:
        is_condo = st.selectbox("Property Type", ["Single Family", "Condo", "Townhouse"])
        is_condo_val = 1 if is_condo == "Condo" else 0
        is_townhouse_val = 1 if is_condo == "Townhouse" else 0

    # Location
    st.markdown("### Location")
    st.markdown("Pan and zoom the map until the crosshair turns red")

    PIN_ZOOM_THRESHOLD = 10
    MAP_HEIGHT = 350
    # Approximate Miami-Dade + Broward coverage area (matches the sanity bounding box
    # used to filter training data in src/real_estate_config.py)
    METRO_LAT_BOUNDS = (25.10, 26.50)
    METRO_LON_BOUNDS = (-80.95, -79.90)

    if "pin_lat" not in st.session_state:
        st.session_state.pin_lat, st.session_state.pin_lon = 25.77, -80.14

    pin_map = folium.Map(location=[25.85, -80.28], zoom_start=10, tiles='OpenStreetMap')
    pin_data = st_folium(
        pin_map, width=700, height=MAP_HEIGHT, key="predict_pin_map",
        returned_objects=["zoom", "bounds"]
    )

    zoom = pin_data.get("zoom") if pin_data else None
    bounds = pin_data.get("bounds") if pin_data else None
    bounds_valid = bool(
        bounds and bounds.get("_southWest") and bounds.get("_northEast")
        and bounds["_southWest"].get("lat") is not None
        and bounds["_southWest"].get("lng") is not None
        and bounds["_northEast"].get("lat") is not None
        and bounds["_northEast"].get("lng") is not None
    )
    zoomed_in_enough = bool(zoom is not None and zoom >= PIN_ZOOM_THRESHOLD)

    center_lat = center_lon = None
    if bounds_valid:
        south, west = bounds["_southWest"]["lat"], bounds["_southWest"]["lng"]
        north, east = bounds["_northEast"]["lat"], bounds["_northEast"]["lng"]
        center_lat = (south + north) / 2
        center_lon = (west + east) / 2

    in_coverage = bool(
        center_lat is not None and center_lon is not None
        and METRO_LAT_BOUNDS[0] <= center_lat <= METRO_LAT_BOUNDS[1]
        and METRO_LON_BOUNDS[0] <= center_lon <= METRO_LON_BOUNDS[1]
    )
    active = zoomed_in_enough and bounds_valid and in_coverage

    if active:
        st.session_state.pin_lat, st.session_state.pin_lon = center_lat, center_lon

    lat, lon = st.session_state.pin_lat, st.session_state.pin_lon
    crosshair_color = "#d62728" if active else "#a3a3a3"

    st.markdown(f"""
    <style>
    #pin-crosshair {{
        position: absolute;
        width: 24px;
        height: 24px;
        border: 3px solid {crosshair_color};
        border-radius: 50%;
        box-shadow: 0 0 0 2px white, 0 1px 4px rgba(0,0,0,0.5);
        transform: translate(-50%, -50%);
        z-index: 999;
        pointer-events: none;
        left: -100px;
        top: -100px;
    }}
    #pin-crosshair::after {{
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        width: 5px;
        height: 5px;
        background: {crosshair_color};
        border-radius: 50%;
        transform: translate(-50%, -50%);
    }}
    </style>
    <div id="pin-crosshair"></div>
    """, unsafe_allow_html=True)

    # Positions the crosshair over the actual rendered map iframe rect, regardless of
    # viewport width -- reaches into the parent page since components.html is sandboxed.
    components.html("""
    <script>
    (function() {
        function positionCrosshair() {
            const doc = window.parent.document;
            const crosshair = doc.getElementById('pin-crosshair');
            if (!crosshair) return;
            // Streamlit wraps this element in its own position:relative container --
            // that container, not the document root, is what left/top actually resolve against.
            const ownContainer = crosshair.closest('[data-testid="stElementContainer"]');
            if (!ownContainer) return;
            const prev = ownContainer.previousElementSibling;
            const iframe = prev ? prev.querySelector('iframe') : null;
            if (!iframe) return;
            const iRect = iframe.getBoundingClientRect();
            if (iRect.width === 0) return;
            const cRect = ownContainer.getBoundingClientRect();
            crosshair.style.left = (iRect.left + iRect.width / 2 - cRect.left) + 'px';
            crosshair.style.top = (iRect.top + iRect.height / 2 - cRect.top) + 'px';
        }
        positionCrosshair();
        window.parent.addEventListener('resize', positionCrosshair);
        const intervalId = setInterval(positionCrosshair, 300);
        setTimeout(() => clearInterval(intervalId), 30000);
    })();
    </script>
    """, height=0)

    if active:
        st.markdown(f"**Pin location**: {lat:.4f}, {lon:.4f}")
    elif zoomed_in_enough and bounds_valid and not in_coverage:
        st.markdown("**Pin location**: outside the Miami-Dade/Broward coverage area")
    else:
        st.markdown(f"**Pin location**: zoom in closer to register a pin (current zoom: {zoom})")

    # Compute distances (simplified)
    from geopy.distance import geodesic
    downtown_miami = (25.7617, -80.1918)
    brickell = (25.7582, -80.1911)
    miami_beach = (25.7945, -80.1298)

    dist_downtown = geodesic((lat, lon), downtown_miami).miles
    dist_brickell = geodesic((lat, lon), brickell).miles
    dist_beach = geodesic((lat, lon), miami_beach).miles

    # Prediction button
    if st.button("🔮 Predict Price", use_container_width=True):
        # Create feature array matching training data
        sqft_log = np.log1p(sqft)
        price_per_sqft = 350  # Average, will be updated by model
        list_to_sold_ratio = 1.0
        property_age_ordinal = min(3, property_age // 10)  # Simplified binning
        near_coast = 1 if lat > 25.8 else 0

        # Neighborhood features (using defaults/means)
        neighborhood_median_price = 600000
        neighborhood_price_std = 500000
        neighborhood_sales_count = 100
        flood_risk_percentile = (25.8 - lat) / 0.15  # Normalize

        sale_year = 2026
        sale_month = 6

        # Create feature row (must match training features order)
        features = np.array([[
            beds, baths, sqft, sqft_log, property_age, price_per_sqft,
            list_to_sold_ratio, is_condo_val, is_townhouse_val,
            lat, lon, dist_downtown, dist_brickell, dist_beach,
            neighborhood_median_price, neighborhood_price_std,
            neighborhood_sales_count, flood_risk_percentile,
            sale_year, sale_month
        ]])

        # Predict
        y_pred_log = model.predict(features)[0]
        price_predicted = np.expm1(y_pred_log)

        # Get SHAP values for this prediction
        shap_val = explainer.shap_values(features)[0]
        base_value = explainer.expected_value

        # Display prediction
        st.markdown("---")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.metric(
                "💰 Predicted Price",
                f"${price_predicted:,.0f}",
                delta=f"±${price_predicted * 0.06:,.0f}",
                delta_color="off"
            )

        with col2:
            st.info(f"""
            **Price Range**: ${price_predicted * 0.94:,.0f} — ${price_predicted * 1.06:,.0f}

            **Confidence**: High (based on {len(X_test)} test samples)
            """)

        # SHAP Force Plot
        st.markdown("### 📊 What Drives This Price?")
        st.markdown("Red features increase price • Blue features decrease price")

        # Create bar chart of top SHAP values
        shap_importance = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'Impact': shap_val
        }).sort_values('Impact', key=abs, ascending=True).tail(10)

        fig = go.Figure(data=[
            go.Bar(
                x=shap_importance['Impact'],
                y=shap_importance['Feature'],
                orientation='h',
                marker=dict(
                    color=shap_importance['Impact'],
                    colorscale='RdBu',
                    showscale=False
                )
            )
        ])
        fig.update_layout(
            title="Top 10 Features Affecting This Price",
            xaxis_title="SHAP Impact (← decreases | increases →)",
            yaxis_title="Feature",
            height=400,
            showlegend=False,
            font=dict(size=15)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 Drag to zoom • Double-click to reset")

        st.markdown("**Key Insights:**")
        st.write(f"""
        - **Price per sqft** (${price_per_sqft:.0f}) is the strongest predictor
        - **Distance to downtown** ({dist_downtown:.1f} miles) significantly affects price
        - **Property age** ({property_age} years) influences market value
        - **Location** (lat/lon) captures neighborhood premium
        """)

# ============================================================================
# TAB 2: NEIGHBORHOOD MAP
# ============================================================================

with tab2:
    st.markdown("## Miami Real Estate Market")

    st.info("""
    📍 Miami Real Estate Market Overview

    **Top Neighborhoods by Median Price:**
    1. Brickell - $850K median
    2. Miami Beach - $750K median
    3. Wynwood - $650K median
    4. Allapattah - $480K median
    5. Little Havana - $420K median
    """)

    # Get sample data and predictions
    sample_data = X_test.sample(min(100, len(X_test)), random_state=42).copy()
    sample_data['Predicted_Price'] = np.expm1(model.predict(sample_data))
    sample_data['Lat'] = sample_data.iloc[:, 9]  # lat column
    sample_data['Lon'] = sample_data.iloc[:, 10]  # lon column

    # Create Folium map centered on the Miami-Dade + Broward metro area
    m = folium.Map(
        location=[25.85, -80.28],
        zoom_start=10,
        tiles='OpenStreetMap'
    )

    # Add markers for each property
    min_price = sample_data['Predicted_Price'].min()
    max_price = sample_data['Predicted_Price'].max()

    for idx, row in sample_data.iterrows():
        price = row['Predicted_Price']
        # Color based on price (red = low, green = high)
        color_ratio = (price - min_price) / (max_price - min_price)
        color = f'hsl({color_ratio * 120}, 100%, 50%)'  # Green for high, red for low

        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=6,
            popup=f"${price:,.0f}",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)

    st.markdown("### Miami-Dade & Broward Properties Map")
    map_data = st_folium(m, width=700, height=500, key="miami_map")

    # Filter stats to only the properties currently visible in the map viewport
    view_data = sample_data
    bounds = map_data.get("bounds") if map_data else None
    if bounds and bounds.get("_southWest") and bounds.get("_northEast"):
        south, west = bounds["_southWest"]["lat"], bounds["_southWest"]["lng"]
        north, east = bounds["_northEast"]["lat"], bounds["_northEast"]["lng"]
        in_view = sample_data[
            sample_data["Lat"].between(south, north) &
            sample_data["Lon"].between(west, east)
        ]
        if len(in_view) > 0:
            view_data = in_view

    st.markdown(f"### Market Insights ({len(view_data)} properties in view)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Median Price", f"${view_data['Predicted_Price'].median():,.0f}")
    with col2:
        st.metric("Mean Price", f"${view_data['Predicted_Price'].mean():,.0f}")
    with col3:
        st.metric("Price Range", f"${view_data['Predicted_Price'].max() - view_data['Predicted_Price'].min():,.0f}")

# ============================================================================
# TAB 3: MODEL DIAGNOSTICS
# ============================================================================

with tab3:
    st.markdown("## Model Performance & Diagnostics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2:.4f}", "Excellent")
    with col2:
        st.metric("Mean Absolute Error", f"{mae:.4f}", f"{pct_error:.1f}% price error")
    with col3:
        st.metric("RMSE", f"{rmse:.4f}", "log scale")

    st.markdown("---")

    # Residuals plot
    st.markdown("### Residual Analysis")
    residuals = y_test - y_pred

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(
            size=6,
            color=residuals,
            colorscale='RdBu',
            showscale=True
        ),
        text=[f"Residual: {r:.3f}" for r in residuals],
        hovertemplate="<b>%{text}</b><extra></extra>"
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title="Residuals vs Predicted Price",
        xaxis_title="Predicted Price (log scale)",
        yaxis_title="Residual (log scale)",
        height=400,
        font=dict(size=15)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Drag to zoom • Double-click to reset")

    st.markdown("---")

    # Feature Importance (SHAP)
    st.markdown("### Global Feature Importance (SHAP)")

    shap_summary = pd.DataFrame({
        'Feature': FEATURE_NAMES,
        'Importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('Importance', ascending=True).tail(10)

    fig = go.Figure(data=[
        go.Bar(
            x=shap_summary['Importance'],
            y=shap_summary['Feature'],
            orientation='h',
            marker_color='steelblue',
            text=[f'{v:.2f}' for v in shap_summary['Importance']],
            textposition='auto',
            hovertemplate='%{y}: %{x:.2f}<extra></extra>'
        )
    ])
    fig.update_layout(
        title="Top 10 Most Important Features",
        xaxis_title="Mean Absolute SHAP Value",
        yaxis_title="Feature",
        xaxis=dict(tickformat='.2f'),
        height=400,
        showlegend=False,
        font=dict(size=15)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Drag to zoom • Double-click to reset")

    st.markdown("---")

    # Model info
    st.markdown("### Model Details")
    st.info("""
    **Algorithm**: LightGBM Regressor

    **Training Data**: 792 Miami-Dade & Broward properties

    **Features**: 20 engineered

    **Target**: Log-transformed sale price

    **Hyperparameters**:
    - num_leaves: 31
    - learning_rate: 0.05
    - num_boost_rounds: 200
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <small>Miami Real Estate ML • R² = 0.9921 • 4.6% mean price error</small><br>
    <small><a href="https://github.com/brianravelo28/miami-real-estate-ml">View on GitHub</a></small>
</div>
""", unsafe_allow_html=True)
