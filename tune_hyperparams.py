"""
Hyperparameter Tuning: Ridge alpha, XGBoost, LightGBM
======================================================
Grid searches all three model classes for all three prediction targets
(Spread, Total, OREB) using the existing walk-forward CV harness.

Reads from data/train_*.csv directly — no build_dataset.py rebuild needed.

Run with Python 3.13:
    /c/Users/Hunter/AppData/Local/Programs/Python/Python313/python tune_hyperparams.py

Expected runtime: 2-4 hours total.
"""

import itertools
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from build_dataset import SPREAD_FEATURES, TOTAL_FEATURES, OREB_FEATURES
from train_and_predict import (
    compute_sample_weights,
    walk_forward_validate,
    train_ridge,
    train_xgboost,
    train_lightgbm,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASELINES = {"Spread": 10.727, "Total": 14.818, "OREB": 4.207}

DATA_DIR = "data"

ALPHA_GRID = [1, 5, 10, 25, 50, 100, 200, 500, 1000]

XGB_GRID = {
    "max_depth":        [3, 4, 5],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 0.8, 1.0],
    "min_child_weight": [3, 5, 10],
}

LGB_GRID = {
    "num_leaves":        [15, 31, 63],
    "min_child_samples": [10, 20, 50],
    "learning_rate":     [0.01, 0.05, 0.1],
}


# ---------------------------------------------------------------------------
# Grid search helpers
# ---------------------------------------------------------------------------

def ridge_grid_search(X, y, w, label):
    """Evaluate all Ridge alpha values for a given target using walk_forward_validate.

    Returns (best_alpha, best_mae).
    """
    results = {}
    print(f"\n  Ridge alpha grid search for {label}:")
    for alpha in ALPHA_GRID:
        # Default-argument binding prevents Python closure bug
        fn = lambda X_tr, y_tr, sample_weight=None, a=alpha: train_ridge(
            X_tr, y_tr, alpha=a, sample_weight=sample_weight
        )
        maes = walk_forward_validate(X, y, fn, sample_weight=w)
        mean_mae = np.mean(maes)
        results[alpha] = mean_mae
        print(f"  Ridge alpha={alpha}: MAE={mean_mae:.3f}", flush=True)

    best_alpha = min(results, key=results.get)
    best_mae = results[best_alpha]
    print(f"  -> Best alpha for {label}: alpha={best_alpha}, MAE={best_mae:.3f}", flush=True)
    return best_alpha, best_mae


def xgb_grid_search(X, y, w, label, objective):
    """Evaluate all XGBoost hyperparameter combos for a given target.

    Returns (best_params_dict, best_mae).
    """
    keys = list(XGB_GRID.keys())
    combos = list(itertools.product(*[XGB_GRID[k] for k in keys]))
    total = len(combos)
    print(f"\n  XGBoost grid search for {label} ({total} combos, objective={objective}):")

    results = {}
    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        # Default-argument binding prevents Python closure bug
        fn = lambda X_tr, y_tr, sample_weight=None, p=params, obj=objective: train_xgboost(
            X_tr, y_tr, objective=obj, sample_weight=sample_weight, **p
        )
        maes = walk_forward_validate(X, y, fn, sample_weight=w)
        mean_mae = np.mean(maes)
        combo_key = tuple(sorted(params.items()))
        results[combo_key] = mean_mae
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"  [{i}/{total}] XGB {param_str}: MAE={mean_mae:.3f}", flush=True)

    best_combo_key = min(results, key=results.get)
    best_mae = results[best_combo_key]
    best_params = dict(best_combo_key)
    param_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
    print(f"  -> Best XGBoost for {label}: {param_str}, MAE={best_mae:.3f}", flush=True)
    return best_params, best_mae


def lgb_grid_search(X, y, w, label, objective):
    """Evaluate all LightGBM hyperparameter combos for a given target.

    Returns (best_params_dict, best_mae).
    """
    keys = list(LGB_GRID.keys())
    combos = list(itertools.product(*[LGB_GRID[k] for k in keys]))
    total = len(combos)
    print(f"\n  LightGBM grid search for {label} ({total} combos, objective={objective}):")

    results = {}
    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        # Default-argument binding prevents Python closure bug
        fn = lambda X_tr, y_tr, sample_weight=None, p=params, obj=objective: train_lightgbm(
            X_tr, y_tr, sample_weight=sample_weight, objective=obj, **p
        )
        maes = walk_forward_validate(X, y, fn, sample_weight=w)
        mean_mae = np.mean(maes)
        combo_key = tuple(sorted(params.items()))
        results[combo_key] = mean_mae
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"  [{i}/{total}] LGB {param_str}: MAE={mean_mae:.3f}", flush=True)

    best_combo_key = min(results, key=results.get)
    best_mae = results[best_combo_key]
    best_params = dict(best_combo_key)
    param_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
    print(f"  -> Best LightGBM for {label}: {param_str}, MAE={best_mae:.3f}", flush=True)
    return best_params, best_mae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print("=" * 70)
    print("NBA HYPERPARAMETER TUNING")
    print("Grid searching: Ridge alpha, XGBoost, LightGBM")
    print("Targets: Spread, Total, OREB")
    print("=" * 70)

    # Load training data (keep as DataFrames — walk_forward_validate uses .iloc)
    df_spread = pd.read_csv(f"{DATA_DIR}/train_spread.csv")
    w_spread = compute_sample_weights(df_spread["GAME_DATE"])
    X_spread = df_spread[SPREAD_FEATURES]
    y_spread = df_spread["SPREAD"]

    df_total = pd.read_csv(f"{DATA_DIR}/train_total.csv")
    w_total = compute_sample_weights(df_total["GAME_DATE"])
    X_total = df_total[TOTAL_FEATURES]
    y_total = df_total["TOTAL"]

    df_oreb = pd.read_csv(f"{DATA_DIR}/train_oreb.csv")
    w_oreb = compute_sample_weights(df_oreb["GAME_DATE"])
    X_oreb = df_oreb[OREB_FEATURES]
    y_oreb = df_oreb["OREB_TOTAL"]

    print(f"\nData loaded:")
    print(f"  Spread training rows: {len(df_spread)}")
    print(f"  Total  training rows: {len(df_total)}")
    print(f"  OREB   training rows: {len(df_oreb)}")

    # Accumulate best results: {(target, model): (best_params_or_alpha, best_mae)}
    best_results = {}

    # -----------------------------------------------------------------------
    # Ridge alpha grid search — all three targets
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("HPT-01/02/03: RIDGE ALPHA GRID SEARCH")
    print("=" * 70)

    ridge_spread_alpha, ridge_spread_mae = ridge_grid_search(X_spread, y_spread, w_spread, "Spread")
    best_results[("Spread", "Ridge")] = (ridge_spread_alpha, ridge_spread_mae)

    ridge_total_alpha, ridge_total_mae = ridge_grid_search(X_total, y_total, w_total, "Total")
    best_results[("Total", "Ridge")] = (ridge_total_alpha, ridge_total_mae)

    ridge_oreb_alpha, ridge_oreb_mae = ridge_grid_search(X_oreb, y_oreb, w_oreb, "OREB")
    best_results[("OREB", "Ridge")] = (ridge_oreb_alpha, ridge_oreb_mae)

    # -----------------------------------------------------------------------
    # XGBoost grid search — all three targets
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("HPT-04: XGBOOST GRID SEARCH (81 combos per target)")
    print("=" * 70)

    xgb_spread_params, xgb_spread_mae = xgb_grid_search(
        X_spread, y_spread, w_spread, "Spread", objective="reg:squarederror"
    )
    best_results[("Spread", "XGBoost")] = (xgb_spread_params, xgb_spread_mae)

    xgb_total_params, xgb_total_mae = xgb_grid_search(
        X_total, y_total, w_total, "Total", objective="reg:squarederror"
    )
    best_results[("Total", "XGBoost")] = (xgb_total_params, xgb_total_mae)

    xgb_oreb_params, xgb_oreb_mae = xgb_grid_search(
        X_oreb, y_oreb, w_oreb, "OREB", objective="count:poisson"
    )
    best_results[("OREB", "XGBoost")] = (xgb_oreb_params, xgb_oreb_mae)

    # -----------------------------------------------------------------------
    # LightGBM grid search — all three targets
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("HPT-05: LIGHTGBM GRID SEARCH (27 combos per target)")
    print("=" * 70)

    lgb_spread_params, lgb_spread_mae = lgb_grid_search(
        X_spread, y_spread, w_spread, "Spread", objective="regression"
    )
    best_results[("Spread", "LightGBM")] = (lgb_spread_params, lgb_spread_mae)

    lgb_total_params, lgb_total_mae = lgb_grid_search(
        X_total, y_total, w_total, "Total", objective="regression"
    )
    best_results[("Total", "LightGBM")] = (lgb_total_params, lgb_total_mae)

    lgb_oreb_params, lgb_oreb_mae = lgb_grid_search(
        X_oreb, y_oreb, w_oreb, "OREB", objective="poisson"
    )
    best_results[("OREB", "LightGBM")] = (lgb_oreb_params, lgb_oreb_mae)

    # -----------------------------------------------------------------------
    # Final results table
    # -----------------------------------------------------------------------
    duration = time.time() - t_start
    print("\n" + "=" * 70)
    print("=== HYPERPARAMETER TUNING RESULTS ===")
    print("=" * 70)
    print(f"\nBaselines: Spread={BASELINES['Spread']}, Total={BASELINES['Total']}, OREB={BASELINES['OREB']}")
    print()

    header = (
        f"{'Target':<8} | {'Model':<10} | {'Best Params':<46} | "
        f"{'CV MAE':>8} | {'vs Baseline':>12} | {'Decision':<24}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for target in ["Spread", "Total", "OREB"]:
        baseline = BASELINES[target]

        # Baseline row
        print(
            f"{target:<8} | {'Ridge':<10} | {'(current baseline)':<46} | "
            f"{baseline:>8.3f} | {'—':>12} | {'current':<24}"
        )

        for model_name in ["Ridge", "XGBoost", "LightGBM"]:
            params_or_alpha, mae = best_results[(target, model_name)]
            delta = mae - baseline

            if model_name == "Ridge":
                params_str = f"alpha={params_or_alpha}"
            else:
                params_str = ", ".join(f"{k}={v}" for k, v in params_or_alpha.items())
            # Truncate long param strings for table display
            if len(params_str) > 46:
                params_str = params_str[:43] + "..."

            delta_str = f"{delta:+.3f}"
            if mae < baseline:
                decision = f"KEPT (delta={delta:+.3f})"
            else:
                decision = f"NO CHANGE (delta={delta:+.3f})"

            print(
                f"{target:<8} | {model_name:<10} | {params_str:<46} | "
                f"{mae:>8.3f} | {delta_str:>12} | {decision:<24}"
            )
        print(sep)

    print(f"\nTotal runtime: {duration/60:.1f} minutes")
    print("\nDone.")


if __name__ == "__main__":
    main()
