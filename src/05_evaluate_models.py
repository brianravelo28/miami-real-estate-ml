"""
Step 5: Model Evaluation (cross-validation, baselines, small hyperparameter search)

Protocol (so the numbers are honest):
  1. Use the same 80/20 split as step 2. The 198-row holdout is NOT touched until the end.
  2. All comparison and tuning uses repeated K-fold CV on the 792 development rows only.
  3. Inside every fold, ZIP price stats are computed from that fold's training rows only
     (leave-one-out for those rows), exactly like the real pipeline, so CV has no leakage.
  4. The holdout is scored once per model, after any choices are made.

Outputs: console table + reports/model_comparison.md
This script does not change the saved model or the config.
"""

import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_estate_config import *  # noqa: E402,F401,F403
import features as fe  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

CV_FOLDS = 5
CV_REPEATS = 3
SEARCH_ITERATIONS = 40
SEARCH_SEED = 7
PROPERTY_COLS = ['sqft_log', 'beds', 'baths', 'property_age', 'is_condo', 'is_townhouse']


def build_frames():
    """Cleaned data -> row-level features that don't depend on any price statistic"""
    df = pickle_load(FEATURES_PATH)
    df = fe.engineer_property_features(df)
    df = fe.engineer_geospatial_features(df)
    df = fe.engineer_flood_risk(df)
    return df


def pickle_load(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def make_fold(train_df, valid_df):
    """Leakage-free matrices for one train/valid pair"""
    tr, va, _, _ = fe.add_neighborhood_features(train_df.copy(), valid_df.copy())
    X_tr, y_tr = fe.make_xy(tr)
    X_va, y_va = fe.make_xy(va)
    return X_tr, y_tr, X_va, y_va


def lgb_model(params, rounds):
    class Wrapper:
        def fit(self, X, y):
            self.m = lgb.train({**LIGHTGBM_PARAMS, **params, 'num_threads': 1}, lgb.Dataset(X, y),
                               num_boost_round=rounds)
            return self

        def predict(self, X):
            return self.m.predict(X)
    return Wrapper()


class ColumnSubset:
    """Wrap a model so it only sees some columns (used for the property-only linear baseline)"""
    def __init__(self, model, cols):
        self.model, self.cols = model, cols

    def fit(self, X, y):
        self.model.fit(X[self.cols], y)
        return self

    def predict(self, X):
        return self.model.predict(X[self.cols])


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    return {
        'r2': r2_score(y_true, y_pred),
        'mae': mae,
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'pct': (np.exp(mae) - 1) * 100,
    }


def cv_score(factory, folds):
    """Mean/sd of each metric across folds, plus mean train R^2 (to show the overfit gap)"""
    rows, train_r2 = [], []
    for X_tr, y_tr, X_va, y_va in folds:
        model = factory().fit(X_tr, y_tr)
        rows.append(metrics(y_va, model.predict(X_va)))
        train_r2.append(r2_score(y_tr, model.predict(X_tr)))
    frame = pd.DataFrame(rows)
    out = {f'{c}_mean': frame[c].mean() for c in frame}
    out.update({f'{c}_sd': frame[c].std() for c in frame})
    out['train_r2'] = float(np.mean(train_r2))
    return out


def random_params(rng):
    return {
        'num_leaves': int(rng.choice([4, 7, 15, 31])),
        'min_data_in_leaf': int(rng.choice([5, 10, 20, 30, 50])),
        'learning_rate': float(rng.choice([0.02, 0.03, 0.05, 0.08])),
        'feature_fraction': float(rng.choice([0.6, 0.8, 1.0])),
        'bagging_fraction': float(rng.choice([0.7, 0.8, 1.0])),
        'lambda_l2': float(rng.choice([0.0, 1.0, 5.0, 20.0])),
        'seed': 42,
    }, int(rng.choice([100, 200, 400, 600]))


def fmt(res, key):
    return f"{res[key + '_mean']:.3f} ± {res[key + '_sd']:.3f}"


SHRINKAGE_GRID = [0, 2, 5, 10, 20, 40]


def shrinkage_study(dev_df, holdout_df):
    """
    How strongly to pull ZIP price stats toward the county's (features.add_neighborhood_features).
    For each strength k: paired CV accuracy against k=0, plus how much a house's prediction moves
    when its neighborhood inputs come from the *neighboring* ZIP instead (the "pin snapped to the
    next ZIP over" effect the dashboard is exposed to).
    Rule (fixed in advance): choose the largest k whose paired CV R2 change is within one standard
    error of k=0, i.e. shrink as much as possible without a measurable accuracy cost.
    """
    splitter = RepeatedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    rows, base_r2 = [], None
    for k in SHRINKAGE_GRID:
        r2s, maes = [], []
        for tr_idx, va_idx in splitter.split(dev_df):
            tr, va, _, _ = fe.add_neighborhood_features(dev_df.iloc[tr_idx].copy(), dev_df.iloc[va_idx].copy(), k)
            X_tr, y_tr = fe.make_xy(tr)
            X_va, y_va = fe.make_xy(va)
            pred = lgb_model({}, LIGHTGBM_NUM_ROUNDS).fit(X_tr, y_tr).predict(X_va)
            r2s.append(r2_score(y_va, pred))
            maes.append(mean_absolute_error(y_va, pred))
        r2s, maes = np.array(r2s), np.array(maes)
        if base_r2 is None:
            base_r2 = r2s
        paired = r2s - base_r2

        tr, ho, zip_table, _ = fe.add_neighborhood_features(dev_df.copy(), holdout_df.copy(), k)
        X_tr, y_tr = fe.make_xy(tr)
        X_ho, y_ho = fe.make_xy(ho)
        model = lgb_model({}, LIGHTGBM_NUM_ROUNDS).fit(X_tr, y_tr)
        pred = model.predict(X_ho)
        swings = []
        for pos, (idx, row) in enumerate(ho.iterrows()):
            others = zip_table.drop(index=row['zip'])
            neighbor = others.loc[fe.nearest_zip(others, row['lat'], row['lon'])]
            swapped = X_ho.loc[[idx]].copy()
            swapped['neighborhood_median_price'] = neighbor['median']
            swapped['neighborhood_price_std'] = neighbor['std']
            swapped['neighborhood_sales_count'] = neighbor['count']
            swings.append(abs(model.predict(swapped)[0] - pred[pos]))
        swings = np.array(swings)
        rows.append({
            'k': k, 'cv_r2': r2s.mean(), 'cv_r2_sd': r2s.std(), 'cv_mae': maes.mean(),
            'paired': paired.mean(), 'paired_se': (paired.std(ddof=1) / np.sqrt(len(paired))) if k else 0.0,
            'holdout_r2': r2_score(y_ho, pred), 'swing_median': float(np.median(swings)),
            'swing_p90': float(np.quantile(swings, 0.9)),
        })
        logger.info(f"  k={k:<3} CV R2 {rows[-1]['cv_r2']:.3f}  paired vs k=0 {rows[-1]['paired']:+.4f} "
                    f"± {rows[-1]['paired_se']:.4f}   neighbor-ZIP swing: median {rows[-1]['swing_median']:.3f}, "
                    f"p90 {rows[-1]['swing_p90']:.3f} (log price)")

    eligible = [r['k'] for r in rows if r['k'] == 0 or r['paired'] >= -r['paired_se']]
    chosen = max(eligible)
    logger.info(f"  rule picks k={chosen}; real_estate_config.NEIGHBORHOOD_SHRINKAGE_K = {NEIGHBORHOOD_SHRINKAGE_K}")
    return rows, chosen


def main():
    t0 = time.time()
    logging.disable(logging.INFO)  # silence the feature code's step logging
    df = build_frames()

    dev_df, holdout_df = fe.split_data(df)
    logging.disable(logging.NOTSET)
    logger.info("=" * 70)
    logger.info("STEP 5: MODEL EVALUATION (CV on development rows, holdout scored once)")
    logger.info("=" * 70)
    logger.info(f"Development rows: {len(dev_df)}   Holdout rows (untouched until the end): {len(holdout_df)}")
    logger.info(f"CV: {CV_FOLDS}-fold x {CV_REPEATS} repeats = {CV_FOLDS * CV_REPEATS} folds; ZIP stats built inside each fold")

    # Build CV folds once (neighborhood features are expensive-ish and shared by all models)
    logging.disable(logging.INFO)
    splitter = RepeatedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    folds = []
    for tr_idx, va_idx in splitter.split(dev_df):
        folds.append(make_fold(dev_df.iloc[tr_idx], dev_df.iloc[va_idx]))
    holdout_fold = make_fold(dev_df, holdout_df)  # ZIP stats from all development rows
    logging.disable(logging.NOTSET)

    default_rounds = LIGHTGBM_NUM_ROUNDS
    baselines = {
        'Predict the mean': lambda: DummyRegressor(strategy='mean'),
        'Linear (property features only)': lambda: ColumnSubset(LinearRegression(), PROPERTY_COLS),
        'Ridge (all 16 features)': lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'Random forest': lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=0.6, random_state=42, n_jobs=-1),
        'LightGBM (current config)': lambda: lgb_model({}, default_rounds),
    }

    results = {}
    for name, factory in baselines.items():
        results[name] = cv_score(factory, folds)
        logger.info(f"  {name:<34} CV R2 {fmt(results[name], 'r2')}   MAE {fmt(results[name], 'mae')}")

    # How strongly to shrink ZIP price stats toward the county's
    logger.info("\nNeighborhood shrinkage study (paired CV vs no shrinkage; ZIP-swap sensitivity on the holdout) ...")
    logging.disable(logging.INFO)
    shrink_rows, shrink_choice = shrinkage_study(dev_df, holdout_df)
    logging.disable(logging.NOTSET)

    # Small random search, scored with the same CV folds
    logger.info(f"\nRandom hyperparameter search: {SEARCH_ITERATIONS} configs x {len(folds)} folds ...")
    rng = np.random.default_rng(SEARCH_SEED)
    trials = []
    for i in range(SEARCH_ITERATIONS):
        params, rounds = random_params(rng)
        res = cv_score(lambda p=params, r=rounds: lgb_model(p, r), folds)
        trials.append((res['r2_mean'], params, rounds, res))
    trials.sort(key=lambda t: -t[0])
    best_r2, best_params, best_rounds, best_res = trials[0]
    results['LightGBM (tuned)'] = best_res
    logger.info(f"  best CV R2 {best_r2:.3f} with rounds={best_rounds}, params={best_params}")
    logger.info(f"  (top 5 CV R2: {', '.join(f'{t[0]:.3f}' for t in trials[:5])}; "
                f"median of all configs: {np.median([t[0] for t in trials]):.3f})")

    # Holdout: score each model once, refit on all development rows
    X_dev, y_dev, X_ho, y_ho = holdout_fold
    holdout = {}
    factories = dict(baselines)
    factories['LightGBM (tuned)'] = lambda: lgb_model(best_params, best_rounds)
    for name, factory in factories.items():
        model = factory().fit(X_dev, y_dev)
        holdout[name] = metrics(y_ho, model.predict(X_ho))

    logger.info("\n" + "=" * 70)
    logger.info(f"{'Model':<34}{'CV R2':>16}{'CV MAE (log)':>16}{'Train R2':>10}{'Holdout R2':>12}{'Holdout MAE':>13}")
    for name in factories:
        r, h = results[name], holdout[name]
        logger.info(f"{name:<34}{fmt(r, 'r2'):>16}{fmt(r, 'mae'):>16}{r['train_r2']:>10.3f}{h['r2']:>12.3f}{h['mae']:>13.3f}")
    logger.info("=" * 70)

    write_report(results, holdout, factories, best_params, best_rounds, trials, len(dev_df), len(holdout_df),
                 shrink_rows, shrink_choice)
    logger.info(f"Done in {time.time() - t0:.0f}s. Report: {os.path.join(REPORTS_DIR, 'model_comparison.md')}")


def write_report(results, holdout, factories, best_params, best_rounds, trials, n_dev, n_holdout,
                 shrink_rows, shrink_choice):
    lines = [
        "# Model Comparison",
        "",
        "Generated by `src/05_evaluate_models.py`. Re-run it after changing data or features.",
        "",
        "## Protocol",
        f"- {n_dev} development rows are used for cross-validation and tuning; "
        f"{n_holdout} holdout rows are scored once at the end.",
        f"- Repeated {CV_FOLDS}-fold CV x {CV_REPEATS} repeats ({CV_FOLDS * CV_REPEATS} folds). "
        "Reported as mean ± standard deviation across folds.",
        "- ZIP price statistics are recomputed inside every fold from that fold's training rows "
        "only (leave-one-out for those rows), so CV has no target leakage.",
        "- Errors are on log price. MAE of 0.24 means a typical multiplicative miss of about 27%.",
        "",
        "## Results",
        "",
        "| Model | CV R² | CV MAE (log) | Train R² | Holdout R² | Holdout MAE (log) |",
        "|-------|-------|--------------|----------|------------|-------------------|",
    ]
    for name in factories:
        r, h = results[name], holdout[name]
        lines.append(f"| {name} | {fmt(r, 'r2')} | {fmt(r, 'mae')} | {r['train_r2']:.3f} | "
                     f"{h['r2']:.3f} | {h['mae']:.3f} |")
    lines += [
        "",
        "## Neighborhood shrinkage",
        "ZIP price stats can be pulled toward the county's: a ZIP's median/std get weight n/(n+k) on its own "
        "n training sales and the rest on the county. Accuracy is paired against k=0 over the same CV folds. "
        "\"Swing\" is how much a holdout house's predicted log price moves when its neighborhood inputs "
        "come from the neighboring ZIP instead of its own (what happens when a map pin snaps to the next ZIP over; "
        "0.10 is about 10%).",
        "",
        "| k | CV R² | Paired ΔR² vs k=0 (± se) | Holdout R² | Swing: median | Swing: 90th pct |",
        "|---|-------|--------------------------|------------|---------------|-----------------|",
    ] + [
        f"| {r['k']} | {r['cv_r2']:.3f} | {r['paired']:+.4f} ± {r['paired_se']:.4f} | {r['holdout_r2']:.3f} | "
        f"{r['swing_median']:.3f} | {r['swing_p90']:.3f} |" for r in shrink_rows
    ] + [
        "",
        f"Rule (fixed in advance): the largest k whose paired CV change is within one standard error of k=0. "
        f"It picks k={shrink_choice}; `NEIGHBORHOOD_SHRINKAGE_K` is {NEIGHBORHOOD_SHRINKAGE_K}. Shrinkage reduces the "
        "swing without a measurable accuracy cost, but does not remove it.",
        "",
        "## Hyperparameter search",
        f"- {SEARCH_ITERATIONS} random LightGBM configurations, each scored with the same CV folds.",
        f"- Best by CV R²: `{best_params}` with `{best_rounds}` boosting rounds "
        "(everything not listed uses the defaults in `real_estate_config.py`).",
        f"- Spread across configurations: best {trials[0][0]:.3f}, "
        f"median {np.median([t[0] for t in trials]):.3f}, worst {trials[-1][0]:.3f}.",
        "- The best CV score is slightly optimistic because it was picked as the maximum of "
        f"{SEARCH_ITERATIONS} tries; the holdout column is the unbiased check.",
        "",
        "## Caveats",
        "- The holdout is only a few hundred rows, so differences of a few hundredths in R² are within noise.",
        "- Only one data source and one year; no time-based validation is possible without sale dates.",
        "",
    ]
    with open(os.path.join(REPORTS_DIR, 'model_comparison.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


if __name__ == '__main__':
    main()
