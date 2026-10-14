"""
Miami Real Estate Market Explorer - Streamlit dashboard
A descriptive look at ~1,000 Miami-Dade and Broward home sales. No model, no predictions.
"""

import os
import sys

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(APP_DIR, 'src'))
import real_estate_config as cfg  # noqa: E402
import summaries  # noqa: E402

st.set_page_config(page_title="Miami Real Estate Explorer", page_icon="🏠", layout="wide")

st.markdown("""
<style>
html, body, p, span, li, label, input, textarea, select, button, div[class*="css"] {
    font-size: 17px !important;
}
small, [data-testid="stCaptionContainer"], [data-testid="stMetricLabel"] {
    font-size: 15px !important;
}
</style>
""", unsafe_allow_html=True)

CHART_HINT = "💡 Drag to zoom • Double-click to reset"
TYPE_ORDER = ['Single family', 'Condo', 'Townhome', 'Other']
TYPE_COLORS = {'Single family': '#1f77b4', 'Condo': '#ff7f0e', 'Townhome': '#2ca02c', 'Other': '#7f7f7f'}


def money(value):
    """$9.9M / $65K (st.metric values are plain text, so no $ escaping is needed)"""
    if pd.isna(value):
        return "n/a"
    return f"${value / 1e6:.2f}M" if value >= 1e6 else f"${value / 1e3:,.0f}K"


@st.cache_data
def load_sales():
    df = pd.read_csv(cfg.CLEAN_DATA_PATH, dtype={'zip': str})
    return summaries.prepare(df)


try:
    sales = load_sales()
except Exception as e:  # missing file on a fresh clone, etc.
    st.error(f"Could not load {os.path.basename(cfg.CLEAN_DATA_PATH)}: {e}\n\n"
             "Run `python run_pipeline.py` (needs the Kaggle CSV in data/).")
    st.stop()

# ============================================================================
# SIDEBAR FILTERS (apply to every tab)
# ============================================================================

with st.sidebar:
    st.markdown("## Filters")
    counties = st.multiselect("County", sorted(sales['county_name'].unique()),
                              default=sorted(sales['county_name'].unique()))
    types = st.multiselect("Property type", TYPE_ORDER, default=TYPE_ORDER)
    lo, hi = int(sales['lastSoldPrice'].min()), int(sales['lastSoldPrice'].max())
    price_range = st.slider("Sale price", lo, hi, (lo, hi), step=25_000, format="$%d")
    st.caption(f"Data: {len(sales):,} closed sales in 2026 (Kaggle), "
               f"{sales['zip'].nunique()} ZIP codes. Filters apply to all tabs.")

view = sales[sales['county_name'].isin(counties)
             & sales['property_type'].isin(types)
             & sales['lastSoldPrice'].between(*price_range)]

st.markdown("# 🏠 Miami Real Estate Market Explorer")
st.markdown("What ~1,000 recent Miami-Dade and Broward home sales show. "
            "This is a description of past sales, not a price estimator.")

if view.empty:
    st.warning("No sales match these filters.")
    st.stop()

tab_overview, tab_map, tab_drivers, tab_zip, tab_about = st.tabs(
    ["📊 Overview", "🗺️ Map by ZIP", "🔎 What goes with price", "📋 ZIP table", "ℹ️ About the data"])

# ============================================================================
# OVERVIEW
# ============================================================================

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sales", f"{len(view):,}")
    c2.metric("Median price", money(view['lastSoldPrice'].median()))
    c3.metric("Median $/sqft", f"${view['price_per_sqft'].median():,.0f}")
    c4.metric("Median size", f"{view['sqft'].median():,.0f} sqft")

    st.markdown("### Price distribution by property type")
    fig = px.box(view, x='property_type', y='lastSoldPrice', color='property_type', points='outliers',
                 category_orders={'property_type': TYPE_ORDER}, color_discrete_map=TYPE_COLORS, log_y=True,
                 labels={'property_type': '', 'lastSoldPrice': 'Sale price (log scale)'})
    fig.update_layout(showlegend=False, height=420, font=dict(size=15))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Box = middle 50% of sales, line = median. {CHART_HINT}")

    st.markdown("### Counties and property types")
    left, right = st.columns(2)
    by_county = summaries.summarize(view, 'county_name').rename(columns={'county_name': 'County'})
    by_type = summaries.summarize(view, 'property_type').rename(columns={'property_type': 'Type'})
    cols = {'sales': 'Sales', 'median_price': 'Median price', 'median_price_per_sqft': 'Median $/sqft',
            'median_sqft': 'Median sqft'}
    fmt = {'Median price': '${:,.0f}', 'Median $/sqft': '${:,.0f}', 'Median sqft': '{:,.0f}'}
    left.dataframe(by_county.rename(columns=cols).style.format(fmt), hide_index=True, use_container_width=True)
    right.dataframe(by_type.rename(columns=cols).style.format(fmt), hide_index=True, use_container_width=True)

# ============================================================================
# MAP
# ============================================================================

with tab_map:
    st.markdown("### Median sale price by ZIP code")
    zips = summaries.zip_summary(view)
    show_thin = st.checkbox(f"Also show ZIPs with fewer than {cfg.MIN_ZIP_SALES} sales (shaded grey)", value=False)
    shown = zips if show_thin else zips[~zips['thin']]

    m = folium.Map(location=[25.85, -80.28], zoom_start=10, tiles='OpenStreetMap')
    if len(shown):
        pct = shown['median_price'].rank(pct=True)  # rank-based, so a few $5M ZIPs don't wash out the rest
        for i, r in shown.iterrows():
            ratio = pct[i]
            color = '#9e9e9e' if r['thin'] else f'hsl({(1 - ratio) * 120}, 85%, 45%)'
            folium.CircleMarker(
                location=[r['lat'], r['lon']],
                radius=4 + 2 * min(r['sales'], 25) ** 0.5,
                popup=folium.Popup(
                    f"<b>ZIP {r['zip']}</b> ({r['county']})<br>{int(r['sales'])} sales<br>"
                    f"Median ${r['median_price']:,.0f}<br>${r['median_price_per_sqft']:,.0f}/sqft"
                    + ("<br><i>too few sales to trust</i>" if r['thin'] else ""), max_width=240),
                color=color, fill=True, fill_color=color, fill_opacity=0.65, weight=1.5,
            ).add_to(m)
    st_folium(m, width=700, height=520, key="zip_map", returned_objects=[])
    st.caption("One circle per ZIP, placed at the ZIP's center (the data has no street addresses). "
               "Size = number of sales; color runs green (lower median) to red (higher median) by rank among the ZIPs shown. "
               "Click a circle for details.")
    st.info(f"{len(zips)} ZIPs have sales under the current filters; {int(zips['thin'].sum())} of them have fewer "
            f"than {cfg.MIN_ZIP_SALES}. A median of 2 or 3 sales mostly reflects which houses happened to sell.")

# ============================================================================
# WHAT GOES WITH PRICE
# ============================================================================

with tab_drivers:
    st.markdown("### Price against size")
    fig = px.scatter(view, x='sqft', y='lastSoldPrice', color='property_type', log_x=True, log_y=True,
                     category_orders={'property_type': TYPE_ORDER}, color_discrete_map=TYPE_COLORS,
                     hover_data={'zip': True, 'beds': True, 'baths': True, 'yr_built': True},
                     labels={'sqft': 'Living area, sqft (log scale)', 'lastSoldPrice': 'Sale price (log scale)',
                             'property_type': 'Type'}, opacity=0.65)
    fig.update_layout(height=480, font=dict(size=15))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Bigger homes cost more, but at any size prices span a wide range. {CHART_HINT}")

    st.markdown("### Median \\$/sqft by group")
    group = st.radio("Group by", ["Bedrooms", "Build period", "County", "Property type"], horizontal=True)
    col = {'Bedrooms': 'beds', 'Build period': 'build_period', 'County': 'county_name',
           'Property type': 'property_type'}[group]
    data = view.copy()
    if col == 'beds':
        data['beds'] = data['beds'].clip(upper=5).map(lambda b: '5+' if b >= 5 else str(int(b)))
    table = summaries.summarize(data, col)
    min_n = 10
    table = table[table['sales'] >= min_n]
    if table.empty:
        st.info(f"No group has at least {min_n} sales under the current filters.")
    else:
        fig = px.bar(table, x=col, y='median_price_per_sqft', text='sales',
                     labels={col: group, 'median_price_per_sqft': 'Median $/sqft'})
        fig.update_traces(texttemplate='n=%{text}', textposition='outside', marker_color='steelblue')
        fig.update_layout(height=400, font=dict(size=15))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Groups with fewer than {min_n} sales are hidden. n = number of sales in each bar. {CHART_HINT}")
    st.warning("These are patterns in past sales, not causes. Bedrooms, size, type and location move together "
               "(condos are smaller and concentrated in particular areas), so a gap between two bars is not "
               "the effect of that one feature.")

# ============================================================================
# ZIP TABLE
# ============================================================================

with tab_zip:
    st.markdown("### Every ZIP code")
    table = summaries.zip_summary(view)
    only_solid = st.checkbox(f"Hide ZIPs with fewer than {cfg.MIN_ZIP_SALES} sales", value=True, key="solid")
    if only_solid:
        table = table[~table['thin']]
    out = table[['zip', 'county', 'sales', 'median_price', 'median_price_per_sqft', 'median_sqft']].rename(columns={
        'zip': 'ZIP', 'county': 'County', 'sales': 'Sales', 'median_price': 'Median price',
        'median_price_per_sqft': 'Median $/sqft', 'median_sqft': 'Median sqft'})
    st.dataframe(out.style.format({'Median price': '${:,.0f}', 'Median $/sqft': '${:,.0f}',
                                   'Median sqft': '{:,.0f}'}), hide_index=True, use_container_width=True)
    st.caption("Click a column header to sort. Sales = number of sales behind each median.")

# ============================================================================
# ABOUT
# ============================================================================

with tab_about:
    st.markdown("### About the data")
    st.markdown(f"""
- **Source:** Kaggle "Florida Real Estate Sold 2026" (10,893 statewide closed sales). Filtered to Miami-Dade
  and Broward by looking up each ZIP's county, then to sales of \\$50K-\\$10M and 400-10,000 sqft:
  **{len(sales):,} sales in {sales['zip'].nunique()} ZIP codes**.
- **Small ZIPs:** the median ZIP has {int(sales['zip'].value_counts().median())} sales, so ZIP medians are noisy.
  ZIPs under {cfg.MIN_ZIP_SALES} sales are flagged.
- **Location is a ZIP center,** not an address. The file has no coordinates, so the map places each ZIP at its center.
- **No sale dates.** This is one snapshot; it says nothing about trends or seasonality.
- **Not in the data:** HOA fees, taxes, condition, views, lot size, waterfront.
- **Not a random sample** of all sales in these counties, so treat the numbers as a description of this file.
""")
    st.markdown("### Why there is no price predictor")
    st.markdown("""
An earlier version predicted prices with LightGBM. Its first score (R² 0.99) came from features built
from the sale price itself; after fixing that leak the honest score was about R² 0.82, with a typical error near
28% and an 80% error band from about -29% to +63%. Location, the part that makes a Miami price interesting,
was the weakest input: with ZIP-level coordinates and about 7 sales per ZIP, the same house priced
differently depending on which neighboring ZIP it landed in (about 10% typically, close to 30% in the worst tenth of cases), and how well a location finds its true ZIP couldn't be validated.
A tool that implies more precision than 990 sales can support was removed instead of caveated further.
""")
    st.markdown("[Code and write-up on GitHub](https://github.com/brianravelo28/miami-real-estate-ml)")
