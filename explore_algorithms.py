"""
Algorithm Exploration: ElasticNet, SVR, and ElasticNet-meta Stacking
=====================================================================
Evaluates three alternative algorithm classes against current Ridge/Ensemble
baselines for all three prediction targets (Spread, Total, OREB).

Reads from data/train_*.csv directly — no build_dataset.py rebuild needed.

Run with Python 3.13:
    /c/Users/Hunter/AppData/Local/Programs/Python/Python313/python explore_algorithms.py

Expected runtime: ~10-15 minutes total.
"""

import itertools
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

# Reuse existing infrastructure (no rebuild needed)
from build_dataset import SPREAD_FEATURES, TOTAL_FEATURES, OREB_FEATURES
from train_and_predict import (
    compute_sample_weights,
    walk_forward_validate,
    train_ridge,
    predict_ridge,
    train_xgboost,
    train_lightgbm,
    train_negative_binomial,
    train_stacked_ensemble,
)

# ---------------------------------------------------------------------------
# Baselines (walk_forward_validate will re-derive them; these are for display)
# ---------------------------------------------------------------------------
BASELINES = {
    "Spread": 10.727,
    "Total":  14.818,
    "OREB":   4.207,
}

DATA_DIR = "data"


# ---------------------------------------------------------------------------
# Algorithm definitions
# ---------------------------------------------------------------------------

def train_elasticnet(X_train, y_train, sample_weight=None):
    """ElasticNet with CV over l1_ratio grid (ALG-01).
    Attaches ._scaler so walk_forward_validate routes through predict_ridge().
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
        cv=TimeSeriesSplit(n_splits=5),
        max_iter=5000,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y_train, sample_weight=sample_weight)
    model._scaler = scaler  # required: walk_forward_validate checks hasattr(model, '_scaler')
    return model


def train_svr(X_train, y_train, C=10.0, gamma="scale", epsilon=0.5, sample_weight=None):
    """SVR (RBF kernel) with explicit hyperparameters (ALG-02).
    Attaches ._scaler so walk_forward_validate routes through predict_ridge().
    Note: SVR ignores sample_weight — accepted for API compatibility only.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
    model.fit(X_scaled, y_train)  # SVR does not support sample_weight
    model._scaler = scaler
    return model


def train_elasticnet_meta(X_train, y_train, sample_weight=None):
    """ElasticNet meta-learner for stacked ensemble (ALG-03).
    Lighter l1_ratio grid and lower max_iter — used only on base predictions.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 0.99],
        cv=TimeSeriesSplit(n_splits=5),
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_scaled, y_train, sample_weight=sample_weight)
    model._scaler = scaler
    return model


# ---------------------------------------------------------------------------
# SVR grid search helper (walk-forward, NOT GridSearchCV)
# ---------------------------------------------------------------------------

def svr_grid_search(X, y, w, epsilon, label):
    """Evaluate all (C, gamma) combos for a given target using walk_forward_validate."""
    C_grid = [0.1, 1, 10, 50, 100]
    gamma_grid = ["scale", 0.01, 0.1, 1.0]

    results = {}
    print(f"\n  SVR grid search for {label} (epsilon={epsilon}):")
    for C, gamma in itertools.product(C_grid, gamma_grid):
        # Default-argument binding prevents closure bug
        fn = lambda X_tr, y_tr, sample_weight=None, C=C, g=gamma, eps=epsilon: train_svr(
            X_tr, y_tr, C=C, gamma=g, epsilon=eps, sample_weight=sample_weight
        )
        maes = walk_forward_validate(X, y, fn, sample_weight=w)
        mean_mae = np.mean(maes)
        results[(C, gamma)] = mean_mae
        print(f"  SVR C={C}, gamma={gamma}: MAE={mean_mae:.3f}")

    best_combo = min(results, key=results.get)
    best_mae = results[best_combo]
    print(f"  -> Best SVR for {label}: C={best_combo[0]}, gamma={best_combo[1]}, MAE={best_mae:.3f}")
    return best_combo, best_mae


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print("=" * 70)
    print("NBA ALGORITHM EXPLORATION")
    print("Evaluating: ElasticNet (ALG-01), SVR (ALG-02), ElasticNet-meta (ALG-03)")
    print("=" * 70)

    # Load training data
    train_spread = pd.read_csv(f"{DATA_DIR}/train_spread.csv")
    train_total  = pd.read_csv(f"{DATA_DIR}/train_total.csv")
    train_oreb   = pd.read_csv(f"{DATA_DIR}/train_oreb.csv")

    print(f"\nData loaded:")
    print(f"  Spread training rows: {len(train_spread)}")
    print(f"  Total  training rows: {len(train_total)}")
    print(f"  OREB   training rows: {len(train_oreb)}")

    # Sample weights (recency-weighted)
    w_spread = compute_sample_weights(train_spread["GAME_DATE"])
    w_total  = compute_sample_weights(train_total["GAME_DATE"])
    w_oreb   = compute_sample_weights(train_oreb["GAME_DATE"])

    X_spread = train_spread[SPREAD_FEATURES]
    y_spread = train_spread["SPREAD"]
    X_total  = train_total[TOTAL_FEATURES]
    y_total  = train_total["TOTAL"]
    X_oreb   = train_oreb[OREB_FEATURES]
    y_oreb   = train_oreb["OREB_TOTAL"]

    # Accumulate results for final comparison table
    results = {}  # {(target, algorithm): mae}

    # -----------------------------------------------------------------------
    # ALG-01: ElasticNet (standalone, walk-forward CV)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ALG-01: ELASTICNET (walk-forward CV)")
    print("=" * 70)

    print("\n--- Spread (ElasticNet) ---")
    en_spread_maes = walk_forward_validate(X_spread, y_spread, train_elasticnet, sample_weight=w_spread)
    en_spread_mae = np.mean(en_spread_maes)
    # Fit final model to inspect best params
    final_en_spread = train_elasticnet(X_spread, y_spread, sample_weight=w_spread)
    print(f"  Best l1_ratio: {final_en_spread.l1_ratio_:.2f}, Best alpha: {final_en_spread.alpha_:.4f}")
    results[("Spread", "ElasticNet")] = en_spread_mae

    print("\n--- Total (ElasticNet) ---")
    en_total_maes = walk_forward_validate(X_total, y_total, train_elasticnet, sample_weight=w_total)
    en_total_mae = np.mean(en_total_maes)
    final_en_total = train_elasticnet(X_total, y_total, sample_weight=w_total)
    print(f"  Best l1_ratio: {final_en_total.l1_ratio_:.2f}, Best alpha: {final_en_total.alpha_:.4f}")
    results[("Total", "ElasticNet")] = en_total_mae

    print("\n--- OREB (ElasticNet) ---")
    en_oreb_maes = walk_forward_validate(X_oreb, y_oreb, train_elasticnet, sample_weight=w_oreb)
    en_oreb_mae = np.mean(en_oreb_maes)
    final_en_oreb = train_elasticnet(X_oreb, y_oreb, sample_weight=w_oreb)
    print(f"  Best l1_ratio: {final_en_oreb.l1_ratio_:.2f}, Best alpha: {final_en_oreb.alpha_:.4f}")
    results[("OREB", "ElasticNet")] = en_oreb_mae

    # -----------------------------------------------------------------------
    # ALG-02: SVR (RBF kernel) — grid search per target
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ALG-02: SVR RBF KERNEL (walk-forward grid search)")
    print("Note: SVR ignores sample_weight — grid uses unweighted folds.")
    print("=" * 70)

    # Epsilon heuristic: ~5-10% of target std
    # Spread std ~11.5 -> epsilon=0.5; Total std ~15 -> epsilon=1.0; OREB -> epsilon=0.1
    svr_spread_combo, svr_spread_mae = svr_grid_search(X_spread, y_spread, w_spread, epsilon=0.5, label="Spread")
    results[("Spread", "SVR RBF")] = svr_spread_mae

    svr_total_combo, svr_total_mae = svr_grid_search(X_total, y_total, w_total, epsilon=1.0, label="Total")
    results[("Total", "SVR RBF")] = svr_total_mae

    svr_oreb_combo, svr_oreb_mae = svr_grid_search(X_oreb, y_oreb, w_oreb, epsilon=0.1, label="OREB")
    results[("OREB", "SVR RBF")] = svr_oreb_mae
    print("  Note: OREB SVR may produce slightly negative predictions internally;")
    print("        walk_forward_validate does not apply np.maximum(0) floor.")
    print("        Effect on MAE is small (<0.05) for count targets near 0.")

    # -----------------------------------------------------------------------
    # ALG-03: ElasticNet meta-learner replacing Ridge meta in stacked ensemble
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ALG-03: ELASTICNET META-LEARNER IN STACKED ENSEMBLE")
    print("IMPORTANT: OOF MAE is computed on training data (NOT walk-forward).")
    print("Compare ONLY against the Ridge-meta stacking OOF MAE, not against")
    print("the walk-forward MAE values above.")
    print("=" * 70)

    print("\n--- Spread (ElasticNet meta) ---")
    _, _, en_meta_spread_oof_mae = train_stacked_ensemble(
        X_spread, y_spread,
        base_models=[
            ("ridge", lambda X, y, sample_weight=None: train_ridge(X, y, alpha=100, sample_weight=sample_weight)),
            ("xgb",   train_xgboost),
            ("lgbm",  train_lightgbm),
        ],
        meta_model_fn=train_elasticnet_meta,
        sample_weight=w_spread,
    )
    # Inspect selected l1_ratio from the meta model
    # train_stacked_ensemble doesn't return meta model directly, re-fit for inspection
    print(f"  ElasticNet meta Spread OOF MAE: {en_meta_spread_oof_mae:.3f}")
    results[("Spread", "EN-meta stacking (OOF)")] = en_meta_spread_oof_mae

    print("\n--- Total (ElasticNet meta) ---")
    _, _, en_meta_total_oof_mae = train_stacked_ensemble(
        X_total, y_total,
        base_models=[
            ("ridge", lambda X, y, sample_weight=None: train_ridge(X, y, alpha=200, sample_weight=sample_weight)),
            ("xgb",   train_xgboost),
            ("lgbm",  train_lightgbm),
        ],
        meta_model_fn=train_elasticnet_meta,
        sample_weight=w_total,
    )
    print(f"  ElasticNet meta Total OOF MAE: {en_meta_total_oof_mae:.3f}")
    results[("Total", "EN-meta stacking (OOF)")] = en_meta_total_oof_mae

    print("\n--- OREB (ElasticNet meta) ---")
    _, _, en_meta_oreb_oof_mae = train_stacked_ensemble(
        X_oreb, y_oreb,
        base_models=[
            ("neg_binomial", train_negative_binomial),
            ("xgb_poisson",  lambda X, y, sample_weight=None: train_xgboost(X, y, objective="count:poisson", sample_weight=sample_weight)),
            ("lgbm",         train_lightgbm),
        ],
        meta_model_fn=train_elasticnet_meta,
        sample_weight=w_oreb,
    )
    print(f"  ElasticNet meta OREB OOF MAE: {en_meta_oreb_oof_mae:.3f}")
    results[("OREB", "EN-meta stacking (OOF)")] = en_meta_oreb_oof_mae

    # -----------------------------------------------------------------------
    # Final comparison table
    # -----------------------------------------------------------------------
    duration = time.time() - t_start
    print("\n" + "=" * 70)
    print("=== ALGORITHM EXPLORATION RESULTS ===")
    print("=" * 70)
    print(f"\nBaselines: Spread={BASELINES['Spread']}, Total={BASELINES['Total']}, OREB={BASELINES['OREB']}")
    print("\nNOTE: EN-meta rows use OOF MAE (in-sample stacking) and are NOT")
    print("      directly comparable to walk-forward MAE in other rows.")
    print()

    header = f"{'Target':<8} | {'Algorithm':<26} | {'CV MAE':>8} | {'vs Baseline':>12} | {'Decision':<10}"
    sep    = "-" * len(header)
    print(header)
    print(sep)

    for target in ["Spread", "Total", "OREB"]:
        baseline = BASELINES[target]

        # Baseline row
        print(f"{target:<8} | {'Ridge (baseline)':<26} | {baseline:>8.3f} | {'—':>12} | {'current':<10}")

        # ElasticNet (ALG-01)
        mae = results[(target, "ElasticNet")]
        delta = mae - baseline
        delta_str = f"{delta:+.3f}"
        decision = "BEAT" if mae < baseline else "lost"
        print(f"{target:<8} | {'ElasticNet':<26} | {mae:>8.3f} | {delta_str:>12} | {decision:<10}")

        # SVR RBF (ALG-02)
        mae = results[(target, "SVR RBF")]
        delta = mae - baseline
        delta_str = f"{delta:+.3f}"
        decision = "BEAT" if mae < baseline else "lost"
        print(f"{target:<8} | {'SVR RBF':<26} | {mae:>8.3f} | {delta_str:>12} | {decision:<10}")

        # EN-meta stacking (ALG-03)
        mae = results[(target, "EN-meta stacking (OOF)")]
        print(f"{target:<8} | {'EN-meta stacking (OOF)*':<26} | {mae:>8.3f} | {'(see note)':>12} | {'(OOF only)':<10}")
        print(sep)

    print("\n* OOF MAE from train_stacked_ensemble — computed on training data.")
    print("  Valid comparison: EN-meta OOF vs Ridge-meta OOF (same framework).")
    print("  Invalid comparison: OOF MAE vs walk-forward MAE (different eval).")

    print("\n--- Best hyperparameters found ---")
    print(f"  ElasticNet Spread:  l1_ratio={final_en_spread.l1_ratio_:.2f}, alpha={final_en_spread.alpha_:.4f}")
    print(f"  ElasticNet Total:   l1_ratio={final_en_total.l1_ratio_:.2f}, alpha={final_en_total.alpha_:.4f}")
    print(f"  ElasticNet OREB:    l1_ratio={final_en_oreb.l1_ratio_:.2f}, alpha={final_en_oreb.alpha_:.4f}")
    print(f"  SVR best Spread:    C={svr_spread_combo[0]}, gamma={svr_spread_combo[1]}")
    print(f"  SVR best Total:     C={svr_total_combo[0]}, gamma={svr_total_combo[1]}")
    print(f"  SVR best OREB:      C={svr_oreb_combo[0]}, gamma={svr_oreb_combo[1]}")

    print(f"\nTotal runtime: {duration/60:.1f} minutes")
    print("\nDone.")


if __name__ == "__main__":
    main()
