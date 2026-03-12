# NBA Game Prediction — Model Optimization

## What This Is

An existing NBA game prediction system that forecasts three per-game targets — point spread, total score, and offensive rebounds (OREB) — for upcoming games listed in `Predictions_filled.csv`. The system pulls data from the NBA API, engineers rolling features, and trains per-target models. This project is a systematic effort to push each target's CV MAE below its current baseline by exploring new algorithms, features, and hyperparameter configurations — with each target optimized independently.

## Core Value

Each target (Spread, Total, OREB) achieves the lowest possible CV MAE using its own best model, features, and parameters — with no forced sharing between targets.

## Requirements

### Validated

- ✓ Walk-forward CV MAE is the decision metric — existing
- ✓ Pipeline: `build_dataset.py` → `train_and_predict.py` — existing
- ✓ Spread baseline: Ridge (alpha=100), CV MAE 10.727 — existing
- ✓ Total baseline: Ensemble (Ridge+XGB+LGB), CV MAE 14.818 — existing
- ✓ OREB baseline: Ensemble (Ridge+XGB+LGB), CV MAE 4.207 — existing
- ✓ Per-target feature sets and rolling windows — existing

### Active

- [ ] Evaluate ElasticNet (L1+L2) as a Spread/Total/OREB alternative — tests mix of Ridge and LASSO
- [ ] Evaluate SVR (RBF kernel) per target — captures nonlinear relationships
- [ ] Evaluate per-target stacking with ElasticNet meta-learner
- [ ] Systematic hyperparameter search (GridSearchCV) on current model params per target
- [ ] Feature selection: LASSO-path to identify zero-weight features and prune/replace
- [ ] New feature candidates: opponent-adjusted ORTG/DRTG diffs, home/away split stats, SRS-style ratings
- [ ] Evaluate quantile regression for Total (median may outperform mean)
- [ ] Update improvements.md with all results (kept AND rejected per target)

### Out of Scope

- Adding a 4th base model (e.g. CatBoost) to ensemble — tested pattern shows ceiling
- N_LATE tuning — N_LATE=10 confirmed optimal; do not test 12, 13, or 15
- Rolling window size changes — already coordinate-descent optimized
- Elo parameter changes — already coordinate-descent optimized
- COVID-era seasons (2019-20, 2020-21) — adds noise, explicitly excluded
- Game-level team factors (REST_DIFF, AVAILABILITY_DIFF) for Total/OREB — symmetric cancellation proven
- Deeper XGB/LightGBM HPT via Optuna — previously caused overfit to fold boundaries

## Context

The codebase uses a walk-forward cross-validation scheme (not random splits) which is the correct evaluation method for time-series sports data. improvements.md documents every tested change with results — it is the authoritative log. The existing models are already well-tuned Ridge/Ensemble baselines; marginal gains are expected to be small (< 0.1 MAE) but meaningful for prediction accuracy. Per-target isolation is critical: a change that helps Spread but hurts OREB should be applied only to Spread.

## Constraints

- **Evaluation**: CV MAE only — no held-out test set, no against-the-spread comparison
- **Isolation**: Each target is optimized independently; shared config changes require all three to benefit
- **Pipeline**: Changes must fit within `build_dataset.py` → `train_and_predict.py` structure
- **Reproducibility**: Every experiment result (positive or negative) logged in improvements.md

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Per-target independent optimization | Prevents one target's improvement from being blocked by another's regression | — Pending |
| CV MAE as sole metric | Consistent with existing evaluation framework | ✓ Good |
| improvements.md as experiment log | Single source of truth; prevents retesting rejected ideas | ✓ Good |

---
*Last updated: 2026-03-12 after initialization*
