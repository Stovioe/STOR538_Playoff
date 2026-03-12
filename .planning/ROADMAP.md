# Roadmap: NBA Game Prediction Model Optimization

## Overview

The existing Spread/Total/OREB prediction models are well-tuned Ridge/Ensemble baselines. This roadmap systematically exhausts the remaining improvement space: first testing whether alternative algorithms (ElasticNet, SVR, stacking variants) beat the current architecture per target, then squeezing gains from hyperparameter tuning on the confirmed architecture, then pruning weak features and evaluating new feature candidates. Every result — kept or rejected — is logged in improvements.md. The project ends when each target has its lowest achievable CV MAE given the available data and pipeline.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Algorithm Exploration** - Evaluate ElasticNet, SVR, and stacking alternatives; confirm or replace current model architecture per target
- [ ] **Phase 2: Hyperparameter Tuning** - Systematic grid search over the confirmed model architecture for each target
- [ ] **Phase 3: Feature Refinement** - Prune near-zero-weight features via LASSO path and evaluate new candidate features

## Phase Details

### Phase 1: Algorithm Exploration
**Goal**: The best model architecture for each target is known and applied — not assumed to be the current default
**Depends on**: Nothing (first phase)
**Requirements**: ALG-01, ALG-02, ALG-03, ALG-04, LOG-01, LOG-02, LOG-03
**Success Criteria** (what must be TRUE):
  1. ElasticNet CV MAE is recorded for all three targets with l1_ratio and alpha grid-searched; result compared against current baseline
  2. SVR (RBF) CV MAE is recorded for all three targets with C and kernel width tuned; result compared against current baseline
  3. Per-target stacking with ElasticNet meta-learner is evaluated; MAE compared against current Ridge meta-learner
  4. Each target is running the algorithm that produces the lowest CV MAE (current or replacement)
  5. improvements.md contains a new entry for every test run — algorithm, per-target MAE before/after, decision (kept/rejected)
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md — Create explore_algorithms.py: evaluate ElasticNet, SVR, and ElasticNet meta-learner for all three targets
- [x] 01-02-PLAN.md — Apply winning algorithms to train_and_predict.py; log all Phase 1 results to improvements.md

### Phase 2: Hyperparameter Tuning
**Goal**: The current model parameters for each target are confirmed optimal across a wider search range than originally tested
**Depends on**: Phase 1
**Requirements**: HPT-01, HPT-02, HPT-03, HPT-04, HPT-05
**Success Criteria** (what must be TRUE):
  1. Ridge alpha for Spread is validated against an expanded grid (e.g., [1, 10, 50, 100, 200, 500, 1000]); current value confirmed or improved
  2. Ridge alpha for Total and OREB are each validated against an expanded grid; current values confirmed or improved
  3. XGBoost n_estimators, max_depth, learning_rate, and subsample are grid-searched per target with overfit prevention; best params applied if MAE improves
  4. LightGBM num_leaves, min_child_samples, and learning_rate are grid-searched per target; best params applied if MAE improves
  5. improvements.md records every grid search result with before/after MAE per target
**Plans**: TBD

### Phase 3: Feature Refinement
**Goal**: Each target's feature set contains only features that measurably contribute to lower CV MAE, with any viable new features added
**Depends on**: Phase 2
**Requirements**: FEAT-01, FEAT-02, FEAT-03, FEAT-04
**Success Criteria** (what must be TRUE):
  1. LASSO path analysis is run per target; any feature with near-zero weight whose removal does not raise MAE is pruned
  2. Opponent-adjusted ORTG/DRTG diffs are correlated against games_features.csv (p < 0.05 threshold) and evaluated for Spread and Total if significant
  3. Home/away split rolling stats are correlated against games_features.csv and evaluated for Spread if significant
  4. No new feature is added to build_dataset.py without a passing correlation check (p < 0.05) recorded in improvements.md
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Algorithm Exploration | 2/2 | Complete | 2026-03-12 |
| 2. Hyperparameter Tuning | 0/? | Not started | - |
| 3. Feature Refinement | 0/? | Not started | - |
