# Requirements: NBA Prediction Model Optimization

**Defined:** 2026-03-12
**Core Value:** Each target (Spread, Total, OREB) achieves the lowest possible CV MAE using its own independently tuned model, features, and parameters.

## v1 Requirements

### Algorithm Evaluation

- [x] **ALG-01**: ElasticNet evaluated as replacement/addition for each target independently (l1_ratio and alpha grid-searched)
- [x] **ALG-02**: SVR (RBF kernel) evaluated per target with kernel width and C tuned via CV
- [x] **ALG-03**: Per-target stacking evaluated with ElasticNet meta-learner replacing Ridge meta-learner
- [x] **ALG-04**: Best algorithm per target identified and applied if it beats baseline CV MAE

### Hyperparameter Tuning

- [ ] **HPT-01**: GridSearchCV over alpha for current Ridge/Ensemble Spread model (expand range beyond [1–200])
- [ ] **HPT-02**: GridSearchCV over alpha for current Ridge/Ensemble Total model
- [ ] **HPT-03**: GridSearchCV over alpha for current Ridge/Ensemble OREB model
- [ ] **HPT-04**: XGBoost per-target HPT: n_estimators, max_depth, learning_rate, subsample (constrained to prevent overfit)
- [ ] **HPT-05**: LightGBM per-target HPT: num_leaves, min_child_samples, learning_rate

### Feature Exploration

- [ ] **FEAT-01**: LASSO path analysis to identify near-zero-weight features per target; prune if removal doesn't hurt MAE
- [ ] **FEAT-02**: Opponent-adjusted ORTG/DRTG diffs (SRS-style) evaluated for Spread and Total
- [ ] **FEAT-03**: Home/away split rolling stats (home ORTG vs away ORTG diffs) evaluated for Spread
- [ ] **FEAT-04**: Correlation check on any new feature candidates against games_features.csv before full rebuild

### Results Logging

- [x] **LOG-01**: improvements.md updated with every experiment result (positive or negative) per target
- [x] **LOG-02**: Each kept change documented with: what changed, before/after MAE, which targets affected
- [x] **LOG-03**: Each rejected change documented with: what was tested, result, reason rejected

## v2 Requirements

### Advanced Methods (deferred)

- **ADV-01**: Quantile regression for Total (median vs mean) — explore after linear model ceiling is clearer
- **ADV-02**: Bayesian optimization (non-Optuna) for XGB/LGB HPT if constrained grid search proves insufficient
- **ADV-03**: Feature importance-weighted ensemble weighting (dynamic meta-learner)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Optuna HPT for XGB/LGB | Previously caused overfit to fold boundaries — proven failure |
| N_LATE tuning | N_LATE=10 confirmed optimal; 15 made all models worse |
| Rolling window size changes | Coordinate-descent optimized; ceiling confirmed |
| Elo parameter changes | Coordinate-descent optimized |
| COVID seasons (2019-20, 2020-21) | Adds noise — explicitly excluded |
| REST_DIFF / AVAILABILITY_DIFF → Total/OREB | Symmetric cancellation proven (p > 0.3) |
| 4th base model (CatBoost) in ensemble | Prior testing showed ceiling on ensemble architecture |
| Short-window defensive diffs for Spread | Tested; < 0.002 delta, model saturated |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ALG-01 | Phase 1 | Complete (01-01) |
| ALG-02 | Phase 1 | Complete (01-01) |
| ALG-03 | Phase 1 | Complete (01-01) |
| ALG-04 | Phase 1 | Complete (01-02) |
| LOG-01 | Phase 1 | Complete (01-01/02) |
| LOG-02 | Phase 1 | Complete (01-02) |
| LOG-03 | Phase 1 | Complete (01-01/02) |
| HPT-01 | Phase 2 | Pending |
| HPT-02 | Phase 2 | Pending |
| HPT-03 | Phase 2 | Pending |
| HPT-04 | Phase 2 | Pending |
| HPT-05 | Phase 2 | Pending |
| FEAT-01 | Phase 3 | Pending |
| FEAT-02 | Phase 3 | Pending |
| FEAT-03 | Phase 3 | Pending |
| FEAT-04 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-12*
*Last updated: 2026-03-12 — ALG-04, LOG-02 marked complete after 01-02 execution; Phase 1 fully closed*
