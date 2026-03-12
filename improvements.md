# NBA Prediction — Baseline & Improvement Candidates

## Current Baseline (as of March 10, 2026)

### Model Performance (Walk-Forward CV MAE)
| Target | Ridge | XGBoost | LightGBM | Ensemble | **Used** |
|--------|-------|---------|----------|----------|---------|
| Spread | **10.727** | 10.942 | 10.919 | 10.765 | Ridge |
| Total  | 14.926 | — | 15.179 | **14.817** | Ensemble |
| OREB   | 4.247 (NB GLM) | 4.313 | 4.287 | **4.207** | Ensemble |

### Current Configuration
```
SEASONS:         2021-22 through 2025-26 (5 seasons, ~5,176 training games)
ROLLING_LONG:    40 games  (columns labeled _40)
ROLLING_SHORT:   7 games   (columns labeled _7)
ELO_K:           16
ELO_HOME_BONUS:  150
ELO_CARRYOVER:   0.5
N_LATE:          10  (last 10 games per team per completed season removed)
Sample weights:  2025-26=1.0, 2024-25=0.5, 2023-24=0.25, 2022-23=0.125, 2021-22=0.0625
SPREAD_RIDGE_ALPHA:      100
TOTAL_RIDGE_ALPHA:       200
TOTAL_META_ALPHA:        10   (negligible effect, ≈0.5)
```

### Current Feature Sets
**Spread (18 features):**
ELO_DIFF, DIFF_EFG_40, DIFF_TOV_RATE_40, DIFF_OREB_RATE_40, DIFF_FT_RATE_40,
DIFF_DEF_EFG_40, DIFF_DEF_TOV_RATE_40, DIFF_DEF_OREB_RATE_40, DIFF_DEF_FT_RATE_40,
DIFF_NET_RTG_40, DIFF_NET_RTG_7, DIFF_PTS_7, DIFF_WIN_PCT, DIFF_HOME_ROAD_WIN_PCT,
REST_DIFF, HOME_IS_B2B, AWAY_IS_B2B, AVAILABILITY_DIFF

**Total (11 features):**
EXPECTED_PACE, SUM_ORTG_40, SUM_DRTG_40, SUM_EFG_40, SUM_FT_RATE_40,
SUM_PTS_40, SUM_PTS_7, PACE_X_EFG, SUM_ORTG_7, SUM_DRTG_7, SUM_PACE_7

**OREB (10 features):**
SUM_OREB_RATE_40, OREB_MATCHUP_TOTAL, SUM_EXPECTED_MISSES_40, EXPECTED_PACE,
SUM_OREB_7, SUM_FGA_40, HOME_ROLL_OREB_RATE_40, AWAY_ROLL_OREB_RATE_40,
HOME_ROLL_DREB_RATE_40, AWAY_ROLL_DREB_RATE_40

---

## Tested Changes — DO NOT RETEST

### KEPT: SUM_PACE_7 → Total features + Total Ridge alpha=200
**Result:** Total Ensemble MAE 14.848 → 14.830 (−0.018). Recent 7-game pace
(r=+0.267 vs Total, p<10⁻⁹⁵) captures hot/cold scoring environments better
than the 40-game expected pace. Alpha sweep [0.5→1000] bottomed at 200.
SUM_EFG_7 tested alongside but added no value — skipped.
**Status:** APPLIED in build_dataset.py (SUM_PACE_7 in both feature functions)
and train_and_predict.py (Total Ridge alpha 1.0 → 200, three call sites).

---

### KEPT: SUM_ORTG_7 + SUM_DRTG_7 → Total features
**Result:** Total Ensemble MAE 14.875 → 14.848 (−0.027). Short-window offensive
and defensive ratings capture recent form beyond what the 40-game baseline
already provides. Pearson r = +0.164/+0.186 vs Total (p < 1e-35).
**Status:** APPLIED in build_dataset.py (both build_model_features and
build_prediction_features). Total now has 10 features.

---

### KEPT: Candidate C — Spread Ridge alpha=100
**Result:** Spread MAE 10.737 → 10.726 (−0.011). Sweet spot found at alpha=100
via sweep of [1, 5, 10, 20, 50, 100, 200]; MAE bottomed at 100 and rose at 200.
**Status:** APPLIED in train_and_predict.py (all three Spread ridge calls updated).

---

### REJECTED: Candidate A — AVAILABILITY_DIFF → Total
**Result:** Pearson r = −0.011, p = 0.38 (not significant). Zero linear relationship
with combined total. Missing players hurt one team's offense but ease the
opponent's defense — effects cancel in the combined total.
**Status:** Do not add. Feature stays in Spread only.

---

### REJECTED: Candidate B — REST_DIFF → Total
**Result:** Pearson r = +0.007, p = 0.59 (not significant). Same cancellation
logic as Candidate A — rest advantage for one team is disadvantage for the other,
sum is unchanged.
**Status:** Do not add. Feature stays in Spread only.

---

### REJECTED: Candidate D — AVAILABILITY_DIFF → OREB
**Result:** Pearson r = +0.013, p = 0.33 (not significant). Roster availability
does not predict combined offensive rebounding.
**Status:** Do not add.

---

### REJECTED: Candidate E — Season progress feature (AVG_GAME_NUM)
**Result:** r=+0.040 vs Total, r=−0.022 vs Spread (not significant). Only ~3 pt
variation in Total across months — rolling windows already capture this. Not
worth the rebuild cost.
**Status:** Do not implement.

---

### REJECTED: Candidate H — Meta-learner alpha tuning (all ensembles)
**Result:** Spread meta: 0.003 delta across [0.1→10] — noise. Total meta: 0.017
delta in quick OOF test but 0.001 in full pipeline (14.817 vs 14.818). OREB meta:
0.001 delta. Total meta changed to alpha=10 (neutral, left in place).
**Status:** All meta-learner alphas effectively insensitive. Not worth further tuning.

---

### REJECTED: Candidate G — Increase N_LATE from 10 to 15
**Result:** All three models got worse. Spread +0.012, Total +0.056, OREB +0.006.
Removed 307 rows (5,176 → 4,869) — data loss outweighs cleaner training signal.
N_LATE=10 is the right balance. Do NOT try 12 or 13 either — the trend is clear.
**Status:** Reverted to N_LATE=10.

---

### REJECTED: Short-window defensive features for Spread
**Result:** DIFF_DEF_EFG_7, DIFF_DEF_TOV_RATE_7, DIFF_DEF_OREB_RATE_7, DIFF_DEF_FT_RATE_7,
DIFF_EFG_7, DIFF_TOV_RATE_7 all tested. Best delta was -0.002 (noise level).
Spread model is saturated on short-window features via DIFF_NET_RTG_7.
**Status:** Do not add any short-window defensive diffs to Spread.

---

### REJECTED: Additional Total interaction features (PACE7_X_EFG7, SUM_NET_RTG_7, SUM_EFG_7)
**Result:** All within 0.002 of BASE11 MAE at alpha=200. SUM_DREB_RATE_40 for
OREB also showed zero improvement (redundant with individual DREB columns).
**Status:** Total and OREB feature sets are at their ceiling.

---

### REJECTED: Candidate F — Drop DIFF_OREB_RATE_40 and DIFF_FT_RATE_40 from Spread
**Result:** Dropping these two features with alpha=100 gave MAE 10.7508 vs
10.7480 keeping all 18. Ridge already shrinks weak features near zero; removing
them adds noise by reducing regularization targets.
**Status:** Keep all 18 Spread features.

---

## Remaining Candidates (Untested)


---

### REJECTED: ElasticNet → Spread / Total / OREB
**Result:** Walk-forward CV MAE vs baselines (explore_algorithms.py, 2026-03-12):
  - Spread: 10.727 → 10.759 (+0.032). Best l1_ratio=0.99, alpha=0.1583 — near-LASSO regime, indistinguishable from Ridge already tuned at alpha=100.
  - Total: 14.818 → 14.866 (+0.048). Best l1_ratio=0.70, alpha=0.5083 — mixed L1/L2, but small alpha means near-OLS; Ridge's heavier regularization already optimal.
  - OREB: 4.207 → 4.235 (+0.028). Best l1_ratio=0.10, alpha=0.2232 — near-Ridge regime; no feature sparsity benefit over the Ensemble.
  Ridge regularization at alpha=100/200 already covers the relevant regularization strength; ElasticNet's L1 penalty adds no sparsity benefit for these feature sets.
**Status:** REJECTED — all three targets lost. Baselines unchanged.

---

### REJECTED: SVR (RBF kernel) → Spread / Total / OREB
**Result:** Walk-forward CV MAE with outer grid search (C=[0.1,1,10,50,100], gamma=['scale',0.01,0.1,1.0], explore_algorithms.py, 2026-03-12):
  - Spread: 10.727 → 10.938 (+0.211); best C=1, gamma=0.01.
  - Total: 14.818 → 14.933 (+0.115); best C=10, gamma=0.01.
  - OREB: 4.207 → 4.232 (+0.025); best C=1, gamma=0.01.
  NBA spread/total/OREB prediction relationships are predominantly linear; the RBF kernel's non-linearity provided no benefit. SVR also ignores sample_weight (recency weighting), which likely accounts for part of the performance gap.
**Status:** REJECTED — all three targets lost. Best hyperparameters retained in train_and_predict.py for model selection but will not be selected by min().

---

### REJECTED: ElasticNet meta-learner in stacking → Spread / Total / OREB
**Result:** Stacking OOF MAE (training data; NOT comparable to walk-forward MAE — compare only within stacking framework):
  - Spread: Ridge meta OOF ≈10.77 → ElasticNet meta OOF 11.003 (+0.23).
  - Total: Ridge meta OOF ≈14.82 → ElasticNet meta OOF 15.744 (+0.92).
  - OREB: Ridge meta OOF ≈4.21 → ElasticNet meta OOF 4.433 (+0.22).
  All three base models (Ridge, XGBoost, LightGBM for Spread/Total; NB GLM, XGBoost Poisson, LightGBM for OREB) contribute useful signal — ElasticNet meta-learner did not zero out any base model weight beneficially; L1 sparsity on a 3-column input provided no advantage over Ridge meta.
**Status:** REJECTED — all three targets lost. Ensemble-EN included in train_and_predict.py model selection dict but will not be selected by min().

---

## How to Test Each Candidate

1. Note the current baseline MAE from the MODEL PERFORMANCE SUMMARY printout.
2. Make only the change described for the candidate.
3. Run: `python build_dataset.py && python train_and_predict.py`
4. Record the new MAE from the MODEL PERFORMANCE SUMMARY printout.
5. Revert if the MAE is worse; keep if it improves. Update this file either way.

---

## What Is Already Near Its Ceiling

- **Ensemble architecture** — Ridge + XGB + LightGBM with Ridge meta-learner is
  a solid setup. Adding a fourth base model (e.g., CatBoost) is unlikely to move
  MAE measurably.
- **Elo parameters** — already optimized via coordinate descent sweep.
- **Rolling window sizes** — already optimized via coordinate descent sweep.
- **More seasons** — going back to 2019-20 or 2020-21 adds COVID-era noise.
- **Deeper hyperparameter tuning on XGBoost/LightGBM** — previously tried with
  Optuna and it made things worse (overfit to fold boundaries).
- **Adding game-level team factors (rest, B2B, win%) to Total/OREB** — tested
  AVAILABILITY_DIFF and REST_DIFF; both p > 0.05 against Total and OREB. The
  symmetric nature of combined targets means individual-team factors cancel out.
- **N_LATE tuning** — N_LATE=15 made all three models worse. N_LATE=10 is optimal;
  do not test 12 or 13, the trend (data loss > signal gain) is clear.
- **Total/OREB interaction features** — PACE7_X_EFG7, SUM_NET_RTG_7, SUM_EFG_7,
  SUM_DREB_RATE_40 all tested; none moved MAE beyond noise (< 0.002 delta).
- **Season progress / game number** — r=0.04 vs Total, not significant vs Spread.
  Negligible signal beyond rolling windows.
- **Ensemble meta-learner alpha** — insensitive for all three models. Sweeps
  showed < 0.002 MAE movement from alpha=0.1 to alpha=10.
- **All remaining original candidates tested** — no further low-hanging fruit
  identified. Model is near its practical ceiling given available features.
