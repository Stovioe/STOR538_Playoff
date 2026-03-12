---
phase: 01-algorithm-exploration
plan: 02
subsystem: model-training
tags: [ElasticNet, SVR, stacking, algorithm-integration, improvements-log]
dependency_graph:
  requires: [01-01-SUMMARY.md, explore_algorithms.py, data/train_spread.csv, data/train_total.csv, data/train_oreb.csv]
  provides: [Updated train_and_predict.py with ElasticNet/SVR/Ensemble-EN selection, improvements.md Phase 1 entries]
  affects: [train_and_predict.py, improvements.md]
tech_stack:
  added: [sklearn.linear_model.ElasticNetCV, sklearn.svm.SVR]
  patterns: [model_maes dict extended with new entries; min() selection auto-picks winner; _scaler attachment for predict_ridge compatibility]
key_files:
  created: []
  modified: [train_and_predict.py, improvements.md]
decisions:
  - All Phase 1 algorithms rejected — all three targets (Spread, Total, OREB) lost vs baselines
  - ElasticNet and SVR added to model selection dicts for completeness; Ridge/Ensemble still selected by min()
  - Baselines confirmed unchanged: Spread Ridge 10.727, Total Ensemble 14.818, OREB Ensemble 4.207
  - Phase 2 starts from same baselines as Phase 1 — no improvement to carry forward
metrics:
  duration_minutes: 8
  completed_date: "2026-03-12"
  tasks_completed: 2
  files_modified: 2
---

# Phase 1 Plan 2: Apply Algorithm Exploration Results Summary

Phase 1 algorithm exploration closed: ElasticNet (standalone), SVR (RBF, grid-searched), and ElasticNet meta-learner in stacking all evaluated via walk-forward CV and all rejected across all three targets; train_and_predict.py extended to include all Phase 1 candidates in model selection dicts, and improvements.md updated with three new REJECTED log entries using exact MAE numbers.

## Algorithm Results (from explore_algorithms.py run, 2026-03-12)

### Applied vs Rejected per Target

| Algorithm | Spread | Total | OREB |
|-----------|--------|-------|------|
| ElasticNet | REJECTED (+0.032) | REJECTED (+0.048) | REJECTED (+0.028) |
| SVR RBF | REJECTED (+0.211) | REJECTED (+0.115) | REJECTED (+0.025) |
| Ensemble-EN (ElasticNet meta) | REJECTED (+0.23 OOF) | REJECTED (+0.92 OOF) | REJECTED (+0.22 OOF) |

**No baseline was beaten. All baselines confirmed for Phase 2.**

### Final CV MAE After Phase 1 (unchanged from Phase 1 start)

| Target | Model | CV MAE |
|--------|-------|--------|
| Spread | Ridge (alpha=100) | 10.727 |
| Total  | Ensemble (Ridge meta) | 14.818 |
| OREB   | Ensemble (NB+XGB Poisson+LightGBM, Ridge meta) | 4.207 |

### Phase 2 Starting Baselines (same as Phase 1)

No targets improved. Phase 2 starts from the same baselines:
- Spread: Ridge alpha=100, CV MAE = 10.727
- Total: Ensemble, CV MAE = 14.818
- OREB: Ensemble, CV MAE = 4.207

## Changes Made

### train_and_predict.py

1. Added imports: `ElasticNetCV` from `sklearn.linear_model`, `SVR` from `sklearn.svm`
2. Added three new functions:
   - `train_elasticnet()`: ElasticNetCV with l1_ratio=[0.1,0.3,0.5,0.7,0.9,0.95,0.99], TimeSeriesSplit(n_splits=5), attaches `_scaler`
   - `train_svr()`: SVR(kernel='rbf') with configurable C/gamma/epsilon, attaches `_scaler`
   - `train_elasticnet_meta()`: ElasticNetCV meta-learner for stacking, lighter grid, attaches `_scaler`
3. Extended each target's model evaluation section with ElasticNet and SVR walk-forward CV calls
4. Added Ensemble-EN stacking (same base models, ElasticNet meta) for each target
5. Extended model_maes dicts (spread, total, oreb) with ElasticNet, SVR, Ensemble-EN entries
6. Added prediction routing for ElasticNet, SVR, Ensemble-EN in all three target sections
7. Applied `np.maximum(oreb_preds, 0)` explicitly for SVR OREB predictions (SVR can predict negative)

### improvements.md

Added three REJECTED entries under "Tested Changes — DO NOT RETEST":
- `### REJECTED: ElasticNet → Spread / Total / OREB` — all deltas positive, Ridge already optimal
- `### REJECTED: SVR (RBF kernel) → Spread / Total / OREB` — linear relationships dominate; SVR no-sample-weight penalty
- `### REJECTED: ElasticNet meta-learner in stacking → Spread / Total / OREB` — all base models contribute; L1 on 3-column input no benefit

## Deviations from Plan

### Auto-completed prerequisite: Plan 01-01 not yet executed

Plan 01-02 depends_on 01-01. The 01-01-SUMMARY.md did not exist, meaning explore_algorithms.py had not been run. The script existed but had not produced documented results.

Action taken: Ran explore_algorithms.py to get actual MAE numbers, then created 01-01-SUMMARY.md with those results. This was a Rule 3 (blocking issue) fix — plan 01-02 requires the Phase 1 MAE values from plan 01-01.

All MAE values in train_and_predict.py and improvements.md are real numbers from the actual script run.

## Self-Check: PASSED

- train_and_predict.py: syntax OK (verified via ast.parse)
- train_and_predict.py contains train_elasticnet(), train_svr(), train_elasticnet_meta() functions
- "ElasticNet" appears 6 times in train_and_predict.py (once per model dict + routing per 3 targets)
- "SVR" appears 6 times in train_and_predict.py
- "Ensemble-EN" appears 6 times in train_and_predict.py
- improvements.md: ElasticNet, SVR, and meta-learner entries all present (verified via Python assertion)
- All entries use real MAE numbers from explore_algorithms.py output
- No placeholder text remains
