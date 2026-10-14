"""
Descriptive summaries used by the dashboard: no model, just medians and counts.

Everything reports the number of sales behind it, because the dataset is small
(about 1,000 sales, a median of 7 per ZIP) and a median of 2 sales is not a neighborhood price.
"""

import pandas as pd

from real_estate_config import MIN_ZIP_SALES

TYPE_LABELS = {
    'single_family': 'Single family',
    'condos': 'Condo',
    'townhomes': 'Townhome',
    'condo_townhome_rowhome_coop': 'Townhome',
    'multi_family': 'Other',
    'mobile': 'Other',
}

AGE_BINS = [0, 1960, 1980, 2000, 2015, 3000]
AGE_LABELS = ['Before 1960', '1960-1979', '1980-1999', '2000-2014', '2015 or newer']


def prepare(df):
    """Add the columns the dashboard groups by: type label, price per sqft, build-year band"""
    out = df.copy()
    out['property_type'] = out['type'].map(TYPE_LABELS).fillna('Other')
    out['price_per_sqft'] = out['lastSoldPrice'] / out['sqft']
    out['build_period'] = pd.cut(out['yr_built'], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    return out


def summarize(df, by):
    """Sales count, median price and median $/sqft for each value of `by` (a column or list)"""
    grouped = df.groupby(by, observed=True)
    table = grouped.agg(
        sales=('lastSoldPrice', 'size'),
        median_price=('lastSoldPrice', 'median'),
        median_price_per_sqft=('price_per_sqft', 'median'),
        median_sqft=('sqft', 'median'),
    )
    return table.reset_index()


def zip_summary(df, min_sales=MIN_ZIP_SALES):
    """One row per ZIP, with a flag for ZIPs too thin to treat as a neighborhood price"""
    table = summarize(df, 'zip')
    geo = df.groupby('zip').agg(lat=('lat', 'first'), lon=('lon', 'first'), county=('county_name', 'first'))
    table = table.merge(geo.reset_index(), on='zip')
    table['thin'] = table['sales'] < min_sales
    return table.sort_values('median_price', ascending=False).reset_index(drop=True)
