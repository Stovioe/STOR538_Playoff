"""
NBA Prediction Models: Training & Prediction
=============================================
Run AFTER build_dataset.py has created the training data.

Usage:
    python train_and_predict.py

Output:
    - Predictions_filled.csv (filled in with Spread, Total, OREB values)
    - models/  (saved model artifacts)
    - results/ (validation MAE reports)
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"


os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

from build_dataset import SPREAD_FEATURES, TOTAL_FEATURES, OREB_FEATURES


def compute_sample_weights(game_dates, decay_per_season=0.5):
    """
    Assign higher weights to more recent games.
    2025-26 → 1.0, 2024-25 → 0.5, 2023-24 → 0.25, 2022-23 → 0.125, 2021-22 → 0.0625.
    """
    dates = pd.to_datetime(game_dates)
    weights = pd.Series(1.0, index=dates.index)
    weights[dates < pd.Timestamp("2025-10-01")] = decay_per_season          # 2024-25
    weights[dates < pd.Timestamp("2024-10-01")] = decay_per_season ** 2     # 2023-24
    weights[dates < pd.Timestamp("2023-10-01")] = decay_per_season ** 3     # 2022-23
    weights[dates < pd.Timestamp("2022-10-01")] = decay_per_season ** 4     # 2021-22
    return weights


# WALK-FORWARD VALIDATION
def walk_forward_validate(X, y, model_fn, n_splits=5, sample_weight=None):
    """
    Evaluate a model using TimeSeriesSplit (walk-forward).
    model_fn: callable that returns a fitted model given X_train, y_train.
    Returns list of MAE scores per fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_train = sample_weight.iloc[train_idx] if sample_weight is not None else None

        model = model_fn(X_train, y_train, sample_weight=w_train)
        if hasattr(model, "_scaler"):
            preds = predict_ridge(model, X_test)
        elif hasattr(model, "model"):  # statsmodels GLM
            preds = predict_nb(model, X_test)
        else:
            preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        maes.append(mae)
        print(f"    Fold {fold+1}: MAE = {mae:.3f}")

    print(f"    Mean MAE: {np.mean(maes):.3f} (+/- {np.std(maes):.3f})")
    return maes


# MODEL DEFINITIONS
def train_ridge(X_train, y_train, alpha=1.0, sample_weight=None):
    """Ridge regression with standardized features."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y_train, sample_weight=sample_weight)
    model._scaler = scaler
    return model


def predict_ridge(model, X):
    X_scaled = model._scaler.transform(X)
    return model.predict(X_scaled)


def train_xgboost(X_train, y_train, objective="reg:squarederror", sample_weight=None, **kwargs):
    """XGBoost regressor with early stopping on a temporal holdout."""
    import xgboost as xgb

    n_val = max(1, int(len(X_train) * 0.15))
    X_tr, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
    y_tr, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]
    w_tr = sample_weight.iloc[:-n_val] if sample_weight is not None else None

    params = {
        "n_estimators": 1000,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "objective": objective,
        "random_state": 42,
        "verbosity": 0,
        "early_stopping_rounds": 50,
    }
    params.update(kwargs)

    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lightgbm(X_train, y_train, sample_weight=None, **kwargs):
    """LightGBM regressor with early stopping on a temporal holdout."""
    import lightgbm as lgb

    n_val = max(1, int(len(X_train) * 0.15))
    X_tr, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
    y_tr, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]
    w_tr = sample_weight.iloc[:-n_val] if sample_weight is not None else None

    params = {
        "n_estimators": 1000,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "random_state": 42,
        "verbose": -1,
    }
    params.update(kwargs)

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
    return model


def train_negative_binomial(X_train, y_train, sample_weight=None):
    """
    Negative Binomial GLM for overdispersed count data (OREB).
    Research shows NB roughly halves prediction error vs. Poisson for NBA counts.
    """
    import statsmodels.api as sm

    X_const = sm.add_constant(X_train, has_constant="add")
    model = sm.GLM(
        y_train,
        X_const,
        family=sm.families.NegativeBinomial(alpha=1.0),
        freq_weights=sample_weight,
    ).fit(disp=False)
    return model


def predict_nb(model, X):
    """Generate predictions from a statsmodels NegativeBinomial GLM."""
    import statsmodels.api as sm
    X_const = sm.add_constant(X, has_constant="add")
    return model.predict(X_const)


# STACKED ENSEMBLE
def train_stacked_ensemble(X_train, y_train, base_models, meta_model_fn, sample_weight=None):
    """
    Train a stacked ensemble.
    base_models: list of (name, train_fn) tuples.
    meta_model_fn: function to train the meta-learner on base predictions.
    Uses KFold OOF predictions to build the meta-learner input.
    """
    from sklearn.model_selection import KFold

    n = len(X_train)
    oof_preds = np.zeros((n, len(base_models)))
    fitted_bases = []

    kf = KFold(n_splits=5, shuffle=False)

    for i, (name, train_fn) in enumerate(base_models):
        print(f"    Training base model: {name}")

        for train_idx, val_idx in kf.split(X_train):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            w_tr = sample_weight.iloc[train_idx] if sample_weight is not None else None

            try:
                m = train_fn(X_tr, y_tr, sample_weight=w_tr)
                if hasattr(m, "_scaler"):
                    oof_preds[val_idx, i] = predict_ridge(m, X_val)
                elif hasattr(m, "model"):  # statsmodels NB
                    oof_preds[val_idx, i] = predict_nb(m, X_val)
                else:
                    oof_preds[val_idx, i] = m.predict(X_val)
            except Exception as exc:
                print(f"      WARNING: fold failed for {name}: {exc}. Using zeros.")
                oof_preds[val_idx, i] = 0.0

        # Retrain on full training data for final predictions
        full_model = train_fn(X_train, y_train, sample_weight=sample_weight)
        fitted_bases.append((name, full_model))

    # Train meta-learner on out-of-fold predictions (weighted by recency)
    meta_X = pd.DataFrame(oof_preds, columns=[name for name, _ in base_models])
    meta_model = meta_model_fn(meta_X, y_train, sample_weight=sample_weight)

    # Compute ensemble OOF MAE for direct comparison against individual models
    ensemble_oof_preds = predict_ridge(meta_model, meta_X)
    oof_mae = mean_absolute_error(y_train, ensemble_oof_preds)
    print(f"    Ensemble OOF MAE: {oof_mae:.3f}")

    return fitted_bases, meta_model, oof_mae


def predict_stacked(fitted_bases, meta_model, X):
    """Generate predictions from a stacked ensemble."""
    base_preds = np.zeros((len(X), len(fitted_bases)))

    for i, (name, model) in enumerate(fitted_bases):
        if hasattr(model, "_scaler"):
            base_preds[:, i] = predict_ridge(model, X)
        elif hasattr(model, "model"):  # statsmodels NB
            base_preds[:, i] = predict_nb(model, X)
        else:
            base_preds[:, i] = model.predict(X)

    meta_X = pd.DataFrame(base_preds, columns=[name for name, _ in fitted_bases])
    if hasattr(meta_model, "_scaler"):
        return predict_ridge(meta_model, meta_X)
    return meta_model.predict(meta_X)


# HYPERPARAMETER TUNING
def tune_xgboost_optuna(X_train, y_train, n_trials=50, objective="reg:squarederror"):
    """
    Bayesian hyperparameter optimization with Optuna.
    Requires: pip install optuna
    """
    import optuna
    import xgboost as xgb

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective": objective,
        }

        tscv = TimeSeriesSplit(n_splits=3)
        maes = []
        for train_idx, val_idx in tscv.split(X_train):
            model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            preds = model.predict(X_train.iloc[val_idx])
            maes.append(mean_absolute_error(y_train.iloc[val_idx], preds))
        return np.mean(maes)

    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best MAE: {study.best_value:.3f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


# MAIN TRAINING AND PREDICTION
def main():
    print("NBA MODEL TRAINING & PREDICTION")

    train_spread = pd.read_csv(f"{DATA_DIR}/train_spread.csv")
    train_total = pd.read_csv(f"{DATA_DIR}/train_total.csv")
    train_oreb = pd.read_csv(f"{DATA_DIR}/train_oreb.csv")
    pred_features = pd.read_csv(f"{DATA_DIR}/prediction_features.csv")

    w_spread = compute_sample_weights(train_spread["GAME_DATE"])
    w_total  = compute_sample_weights(train_total["GAME_DATE"])
    w_oreb   = compute_sample_weights(train_oreb["GAME_DATE"])
    print(f"  Sample weight distribution — 2021-22: 0.0625, 2022-23: 0.125, 2023-24: 0.25, 2024-25: 0.50, 2025-26: 1.00")

    print("\nSPREAD MODEL")

    X_spread = train_spread[SPREAD_FEATURES]
    y_spread = train_spread["SPREAD"]

    print("\n  Ridge baseline:")
    ridge_spread_maes = walk_forward_validate(X_spread, y_spread, lambda X, y, sample_weight=None: train_ridge(X, y, alpha=1.0, sample_weight=sample_weight), sample_weight=w_spread)

    print("\n  XGBoost:")
    xgb_spread_maes = walk_forward_validate(X_spread, y_spread, train_xgboost, sample_weight=w_spread)

    print("\n  LightGBM:")
    lgbm_spread_maes = walk_forward_validate(X_spread, y_spread, train_lightgbm, sample_weight=w_spread)

    print("\n  Training stacked ensemble (final)...")
    spread_bases, spread_meta, spread_oof_mae = train_stacked_ensemble(
        X_spread, y_spread,
        base_models=[
            ("ridge", lambda X, y, sample_weight=None: train_ridge(X, y, alpha=1.0, sample_weight=sample_weight)),
            ("xgb", train_xgboost),
            ("lgbm", train_lightgbm),
        ],
        meta_model_fn=lambda X, y, sample_weight=None: train_ridge(X, y, alpha=0.5, sample_weight=sample_weight),
        sample_weight=w_spread,
    )

    print("  Training final individual models on full data...")
    final_ridge_spread = train_ridge(X_spread, y_spread, alpha=1.0, sample_weight=w_spread)
    final_xgb_spread   = train_xgboost(X_spread, y_spread, sample_weight=w_spread)
    final_lgbm_spread  = train_lightgbm(X_spread, y_spread, sample_weight=w_spread)

    spread_model_maes = {
        "Ridge":    np.mean(ridge_spread_maes),
        "XGBoost":  np.mean(xgb_spread_maes),
        "LightGBM": np.mean(lgbm_spread_maes),
        "Ensemble": spread_oof_mae,
    }
    best_spread = min(spread_model_maes, key=spread_model_maes.get)
    X_pred_spread = pred_features[SPREAD_FEATURES]
    if best_spread == "Ridge":
        spread_preds = predict_ridge(final_ridge_spread, X_pred_spread)
    elif best_spread == "XGBoost":
        spread_preds = final_xgb_spread.predict(X_pred_spread)
    elif best_spread == "LightGBM":
        spread_preds = final_lgbm_spread.predict(X_pred_spread)
    else:
        spread_preds = predict_stacked(spread_bases, spread_meta, X_pred_spread)
    print(f"  Using {best_spread} for Spread predictions (MAE {spread_model_maes[best_spread]:.3f})")
    print(f"  Spread predictions generated: mean={np.mean(spread_preds):.2f}, std={np.std(spread_preds):.2f}")

    print("\nTOTAL MODEL")

    X_total = train_total[TOTAL_FEATURES]
    y_total = train_total["TOTAL"]

    print("\n  Ridge baseline:")
    ridge_total_maes = walk_forward_validate(X_total, y_total, lambda X, y, sample_weight=None: train_ridge(X, y, alpha=1.0, sample_weight=sample_weight), sample_weight=w_total)

    print("\n  LightGBM:")
    lgbm_total_maes = walk_forward_validate(X_total, y_total, train_lightgbm, sample_weight=w_total)

    print("\n  Training stacked ensemble (final)...")
    total_bases, total_meta, total_oof_mae = train_stacked_ensemble(
        X_total, y_total,
        base_models=[
            ("ridge", lambda X, y, sample_weight=None: train_ridge(X, y, alpha=1.0, sample_weight=sample_weight)),
            ("xgb", train_xgboost),
            ("lgbm", train_lightgbm),
        ],
        meta_model_fn=lambda X, y, sample_weight=None: train_ridge(X, y, alpha=0.5, sample_weight=sample_weight),
        sample_weight=w_total,
    )

    print("  Training final individual models on full data...")
    final_ridge_total = train_ridge(X_total, y_total, alpha=1.0, sample_weight=w_total)
    final_lgbm_total  = train_lightgbm(X_total, y_total, sample_weight=w_total)

    total_model_maes = {
        "Ridge":    np.mean(ridge_total_maes),
        "LightGBM": np.mean(lgbm_total_maes),
        "Ensemble": total_oof_mae,
    }
    best_total = min(total_model_maes, key=total_model_maes.get)
    X_pred_total = pred_features[TOTAL_FEATURES]
    if best_total == "Ridge":
        total_preds = predict_ridge(final_ridge_total, X_pred_total)
    elif best_total == "LightGBM":
        total_preds = final_lgbm_total.predict(X_pred_total)
    else:
        total_preds = predict_stacked(total_bases, total_meta, X_pred_total)
    print(f"  Using {best_total} for Total predictions (MAE {total_model_maes[best_total]:.3f})")
    print(f"  Total predictions generated: mean={np.mean(total_preds):.2f}, std={np.std(total_preds):.2f}")

    print("\nOREB MODEL")

    X_oreb = train_oreb[OREB_FEATURES]
    y_oreb = train_oreb["OREB_TOTAL"]

    print("\n  Negative Binomial GLM baseline:")
    ridge_oreb_maes = walk_forward_validate(X_oreb, y_oreb, train_negative_binomial, sample_weight=w_oreb)

    print("\n  XGBoost (Poisson):")
    xgb_oreb_maes = walk_forward_validate(
        X_oreb, y_oreb,
        lambda X, y, sample_weight=None: train_xgboost(X, y, objective="count:poisson", sample_weight=sample_weight),
        sample_weight=w_oreb,
    )

    print("\n  LightGBM:")
    lgbm_oreb_maes = walk_forward_validate(X_oreb, y_oreb, train_lightgbm, sample_weight=w_oreb)

    # Stacked ensemble: NB GLM + XGBoost(Poisson) + LightGBM, per planning doc.
    # NB GLM handles overdispersion in OREB counts better than Ridge as a base.
    print("\n  Training stacked ensemble (final)...")
    oreb_bases, oreb_meta, oreb_oof_mae = train_stacked_ensemble(
        X_oreb, y_oreb,
        base_models=[
            ("neg_binomial", train_negative_binomial),
            ("xgb_poisson", lambda X, y, sample_weight=None: train_xgboost(X, y, objective="count:poisson", sample_weight=sample_weight)),
            ("lgbm", train_lightgbm),
        ],
        meta_model_fn=lambda X, y, sample_weight=None: train_ridge(X, y, alpha=0.5, sample_weight=sample_weight),
        sample_weight=w_oreb,
    )

    print("  Training final individual models on full data...")
    final_nb_oreb   = train_negative_binomial(X_oreb, y_oreb, sample_weight=w_oreb)
    final_xgb_oreb  = train_xgboost(X_oreb, y_oreb, objective="count:poisson", sample_weight=w_oreb)
    final_lgbm_oreb = train_lightgbm(X_oreb, y_oreb, sample_weight=w_oreb)

    oreb_model_maes = {
        "NB GLM":        np.mean(ridge_oreb_maes),
        "XGBoost Pois.": np.mean(xgb_oreb_maes),
        "LightGBM":      np.mean(lgbm_oreb_maes),
        "Ensemble":      oreb_oof_mae,
    }
    best_oreb = min(oreb_model_maes, key=oreb_model_maes.get)
    X_pred_oreb = pred_features[OREB_FEATURES]
    if best_oreb == "NB GLM":
        oreb_preds = predict_nb(final_nb_oreb, X_pred_oreb)
    elif best_oreb == "XGBoost Pois.":
        oreb_preds = final_xgb_oreb.predict(X_pred_oreb)
    elif best_oreb == "LightGBM":
        oreb_preds = final_lgbm_oreb.predict(X_pred_oreb)
    else:
        oreb_preds = predict_stacked(oreb_bases, oreb_meta, X_pred_oreb)
    oreb_preds = np.maximum(oreb_preds, 0)  # floor at 0 — OREB is a count
    print(f"  Using {best_oreb} for OREB predictions (MAE {oreb_model_maes[best_oreb]:.3f})")
    print(f"  OREB predictions generated: mean={np.mean(oreb_preds):.2f}, std={np.std(oreb_preds):.2f}")

    print("\nWRITING PREDICTIONS")

    predictions = pd.read_csv("Predictions.csv")

    predictions["Spread"] = np.round(spread_preds, 1)
    predictions["Total"] = np.round(total_preds, 1)
    predictions["OREB"] = np.round(oreb_preds, 1)

    predictions.to_csv("Predictions_filled.csv", index=False)
    print(f"  Saved Predictions_filled.csv with {len(predictions)} games")

    print(f"\n  SANITY CHECKS:")
    print(f"    Spread  - min: {predictions['Spread'].min():.1f}, max: {predictions['Spread'].max():.1f}, "
          f"mean: {predictions['Spread'].mean():.1f}")
    print(f"    Total   - min: {predictions['Total'].min():.1f}, max: {predictions['Total'].max():.1f}, "
          f"mean: {predictions['Total'].mean():.1f}")
    print(f"    OREB    - min: {predictions['OREB'].min():.1f}, max: {predictions['OREB'].max():.1f}, "
          f"mean: {predictions['OREB'].mean():.1f}")

    for col in ["Spread", "Total", "OREB"]:
        n_null = predictions[col].isna().sum()
        if n_null > 0:
            print(f"    WARNING: {n_null} NaN values in {col}!")

    print("\nMODEL PERFORMANCE SUMMARY")
    summary = [
        {
            "target":   "Spread",
            "used":     best_spread,
            "models":   list(spread_model_maes.items()),
        },
        {
            "target":   "Total",
            "used":     best_total,
            "models":   list(total_model_maes.items()),
        },
        {
            "target":   "OREB",
            "used":     best_oreb,
            "models":   list(oreb_model_maes.items()),
        },
    ]

    for entry in summary:
        mae_lookup = dict(entry["models"])
        model_str = "  |  ".join(
            f"{name}: {mae:.3f}{'*' if name == entry['used'] else ''}"
            for name, mae in entry["models"]
        )
        print(f"\n  {entry['target']}")
        print(f"    {model_str}")
        print(f"    PREDICTIONS USE: {entry['used']}  (MAE {mae_lookup[entry['used']]:.3f})  (* = selected)")

    print("\nDone.")


if __name__ == "__main__":
    main()
