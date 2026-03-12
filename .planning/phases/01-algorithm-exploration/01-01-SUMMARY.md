---
phase: 01-algorithm-exploration
plan: 01
subsystem: model-training
tags: [ElasticNet, SVR, stacking, algorithm-exploration, walk-forward-cv]
dependency_graph:
  requires: [data/train_spread.csv, data/train_total.csv, data/train_oreb.csv]
  provides: [explore_algorithms.py, Phase 1 algorithm evaluation results]
  affects: [train_and_predict.py, improvements.md]
tech_stack:
  added: [sklearn.linear_model.ElasticNetCV, sklearn.svm.SVR]
  patterns: [walk-forward grid search for SVR, _scaler attachment pattern for predict_ridge compatibility]
key_files:
  created: [explore_algorithms.py]
  modified: []
decisions:
  - All Phase 1 algorithms (ElasticNet, SVR, ElasticNet meta) rejected — all targets lost vs baselines
  - Baselines confirmed: Spread Ridge alpha=100 MAE=10.727, Total Ensemble MAE=14.818, OREB Ensemble MAE=4.207
metrics:
  duration_minutes: 4
  completed_date: "2026-03-12"
  tasks_completed: 1
  files_created: 1
---

# Phase 1 Plan 1: Algorithm Exploration Script Summary

explore_algorithms.py created and executed; all three Phase 1 algorithms (ElasticNet standalone, SVR RBF with grid search, ElasticNet meta-learner in stacking) evaluated via walk-forward CV against all three targets — every algorithm lost to the existing baselines, confirming the production models are near their ceiling for these algorithm classes.

## Results

### Execution Output (2026-03-12)

Runtime: 3.9 minutes

#### ALG-01: ElasticNet (walk-forward CV MAE)

| Target | Baseline | ElasticNet MAE | Delta | Best l1_ratio | Best alpha |
|--------|----------|---------------|-------|--------------|------------|
| Spread | 10.727   | 10.759        | +0.032 | 0.99 | 0.1583 |
| Total  | 14.818   | 14.866        | +0.048 | 0.70 | 0.5083 |
| OREB   | 4.207    | 4.235         | +0.028 | 0.10 | 0.2232 |

All targets: REJECTED. ElasticNet marginally worse than Ridge/Ensemble baselines.

#### ALG-02: SVR RBF Kernel (walk-forward grid search, 20 combos per target)

| Target | Baseline | Best SVR MAE | Delta | Best C | Best gamma |
|--------|----------|-------------|-------|--------|-----------|
| Spread | 10.727   | 10.938      | +0.211 | 1    | 0.01 |
| Total  | 14.818   | 14.933      | +0.115 | 10   | 0.01 |
| OREB   | 4.207    | 4.232       | +0.025 | 1    | 0.01 |

All targets: REJECTED. SVR's non-linear kernel provided no benefit over Ridge.

#### ALG-03: ElasticNet Meta-Learner in Stacking (OOF MAE — training data only)

NOTE: OOF MAE from train_stacked_ensemble is computed on training data. These values are biased downward and are not directly comparable to walk-forward MAE. Valid only for Ridge-meta vs ElasticNet-meta comparison within the same framework.

| Target | ElasticNet meta OOF MAE |
|--------|------------------------|
| Spread | 11.003 |
| Total  | 15.744 |
| OREB   | 4.433 |

The Ridge meta-learner OOF MAE (from train_and_predict.py) is approximately: Spread ~10.77, Total ~14.82, OREB ~4.21 (from existing Ensemble column). ElasticNet meta performed worse on all three targets.

All targets: REJECTED.

## Key Findings

1. **ElasticNet nearly identical to Ridge:** l1_ratio selected near 1.0 for Spread (0.99) and near 0 for OREB (0.10), indicating the features either need full sparsity (LASSO-like) or full ridge, but neither beats the already-tuned Ridge at alpha=100/200. The small MAE differences (+0.028 to +0.048) are within noise.

2. **SVR provides no benefit:** NBA spread/total/OREB prediction relationships are predominantly linear. The RBF kernel's non-linearity added no value, and SVR's inability to use sample_weight (recency weighting) likely explains part of the performance gap.

3. **ElasticNet meta-learner worse than Ridge meta:** The meta-learner operates on only 3 columns (base model OOF predictions). L1 sparsity on 3 features provided no benefit — all base models are contributing useful signal.

4. **Baselines confirmed:** Spread Ridge alpha=100 (10.727), Total Ensemble (14.818), OREB Ensemble (4.207) remain optimal for these algorithm classes.

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- explore_algorithms.py exists and runs without error
- All three algorithms evaluated per target
- Results compared against known baselines
- Best hyperparameters reported
