# NBA Playoff Prediction Project: Planning Document

## 1. Data Sources

**Primary Source: `nba_api` Python package (v1.11.4)**
Wraps 120+ undocumented stats.nba.com endpoints. No API key needed. Provides game logs, advanced box scores, and team stats going back to the 1996-97 season. We use a 0.7-second delay between API calls to respect rate limits.

Endpoints used:
- `LeagueGameFinder` — pulls every completed game for a given season with basic box score stats
- `PlayerGameLogs` — bulk per-player per-game stats (minutes, team) for all players in a season; one call per season. Used to compute historical team availability scores.
- `LeagueDashPlayerStats` — season-average per-game stats (MIN, NET_RATING, etc.) per player; used for prediction-time availability when actual game minutes are unknown.

Endpoints originally planned but replaced:
- `TeamDashboardByGeneralSplits` — superseded by rolling averages computed from `LeagueGameFinder` box scores
- `LeagueDashTeamStats` — same reason; all advanced metrics derived in-pipeline
- `BoxScoreAdvancedV3` — replaced by `compute_derived_advanced_stats()`, which estimates Pace, ORtg, DRtg, eFG%, etc. directly from basic box score columns. Avoids ~5,000 individual API calls.

**Injury Source: ESPN Public API**
`https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries`
Returns current NBA injury report (Out/Doubtful players by team). No authentication required. Called once at prediction time to zero out injured players' minute contributions in the availability score.

**Outside Data: Basketball Reference — Player BPM**
Scraped from `basketball-reference.com/leagues/NBA_{year}_advanced.html`. Provides Box Plus/Minus (BPM) for every player in a season. Used as the quality weight in the player availability feature. 3-second delay between requests to respect BBRef rate limits. Traded players are de-duplicated by keeping the "TOT" (season totals) row. Files saved locally as `data/bbref_advanced_{year}.csv` for years 2022–2026.

**Not implemented (future work):**
- ESPN schedule API (`scoreboard?dates=YYYYMMDD`) — schedule pre-populated in `Predictions.csv`
- NBA.com Hustle Stats (`LeagueDashPTStats`) — contested rebounds, loose balls; relevant for OREB but deprioritized


## 2. Data Cleaning Summary

**Step 1: Pull raw game logs.** Use `LeagueGameFinder` for 5 seasons: 2021-22, 2022-23, 2023-24, 2024-25, and 2025-26. Each row is one team's performance in one game. Filter to regular season only.

**Step 2: Reshape to game-level rows.** Each game appears twice in raw data (once per team). Merge on `GAME_ID` to create a single row per game with `HOME_` and `AWAY_` prefixed columns. Identify home/away by checking `MATCHUP` for "vs." (home) or "@" (away).

**Step 3: Compute target variables.**
- `Spread = HOME_PTS - AWAY_PTS`
- `Total = HOME_PTS + AWAY_PTS`
- `OREB = HOME_OREB + AWAY_OREB`

**Step 3b: Player availability scores.** Pull player BPM from BBRef and per-game minutes from `PlayerGameLogs`. For each historical game compute `sum(BPM × rolling_prior_minutes)` per team → `HOME_AVAILABILITY` and `AWAY_AVAILABILITY`. `GAME_DATE` is provided directly by `PlayerGameLogs` and is not re-merged from the game log to avoid column conflicts. Each player's minute contribution uses a 15-game rolling average of prior games (`shift(1)` before rolling) rather than actual game minutes, eliminating look-ahead bias. BPM is taken from the **prior season** (season label shifted forward by one year) to prevent end-of-season leakage: e.g., for a 2024-25 game the model uses 2023-24 BPM. Merged onto the game-level DataFrame before feature engineering.

**Step 4: Handle missing data.** Games with missing box score stats (postponed, incomplete) are dropped. Rolling features require `min_periods=5` — a team's first few games of any window are set to NaN and excluded from training via `dropna`.

**Step 5: Late-season game filtering.** For all completed seasons (not the current 2025-26 season), remove each team's last 10 games. These games are corrupted by load management — star players rest in meaningless late-season games — which introduces noise into rolling statistics. Per-team game sequences are built from both home and away appearances to correctly identify each team's final 10 games. Games where either team is in their last 10 are removed.

**Step 6: Sort chronologically.** All games sorted by `GAME_DATE` ascending. Rolling features computed strictly on prior games using `shift(1)` to prevent look-ahead bias.

**Step 7: Train/test split.** Walk-forward validation: train on earlier seasons, test on 2025-26 through the current date. For final predictions, train on all available data through the last completed game.


## 3. Engineered Variables

### 3A. Elo Rating (per team, updated after each game)

**Formula:**
```
Elo_new = Elo_old + K * margin_mult * (S - E)
where:
  K             = 16
  margin_mult   = min(log(|margin| + 1), 2.5)  -- margin of victory scaling
  S             = 1 if win, 0 if loss
  E             = 1 / (1 + 10^((Elo_opp - Elo_self + home_bonus) / 400))
  home_bonus    = 150 (added to home team's Elo when computing win probability)
```

Seasonal carryover: `Elo_start = 0.50 * Elo_end_prev + 0.50 * 1500`
(50% regression toward 1500 at season start — more aggressive reset vs. prior 25%, preventing stale prior-season Elo from dominating)

**Feature used:** `ELO_DIFF = Home_Elo - Away_Elo + 150`

**Parameter justification (from justify_params.py sweep, n=5171 games):**
- K=16 outperformed K=32 on all 3 models (Spread MAE: 10.836 vs. 10.862). Lower K produces a more stable, less noisy signal; the margin multiplier already handles expressiveness.
- Carryover=0.50 beat 0.75 on Spread and OREB (Spread MAE: 10.853 vs. 10.862).
- Home bonus=150 best for Spread (MAE: 10.858 vs. 10.862 at 100). Better reflects the ~3.5-pt home court advantage in modern NBA.
- Margin multiplier: a 10-pt win scales K by ~2.4×; a 1-pt win scales by ~0.69×. Blowouts shift Elo more aggressively, making dominant teams rise faster.

**Pearson r with Spread: +0.375, p=5.1e-172 (n=5171).** Strongest single predictor in the Spread model.

**Why:** Elo ratings encode cumulative team strength in a single number. FiveThirtyEight demonstrated that standalone Elo achieves 65-68% win prediction accuracy with zero box-score data. The Elo differential is the single best low-dimensional predictor of spread. Cited: Nate Silver, "How We Calculate NBA Elo Ratings," FiveThirtyEight.


### 3B. Offensive Four Factors Differentials (rolling 40-game baseline, 7-game recent)

Dean Oliver's Four Factors capture 96% of variance in team wins. We compute each factor as a rolling average for both teams, then take the differential (home minus away). Two windows are used: a 40-game baseline (column suffix `_30`) and a 7-game recent-form window (column suffix `_10`). The suffix labels are fixed in the codebase; the actual window sizes are configured by `ROLLING_LONG=40` and `ROLLING_SHORT=7`.

| Factor | Formula | Oliver's Weight |
|--------|---------|-----------------|
| Effective FG% | `(FGM + 0.5 * FG3M) / FGA` | 40% |
| Turnover Rate | `TOV / (FGA + 0.44 * FTA + TOV)` | 25% |
| Offensive Rebound Rate | `OREB / (OREB + OPP_DREB)` | 20% |
| Free Throw Rate | `FTM / FGA` | 15% |

**Features used (4 offensive differentials, 40-game window):**
- `DIFF_EFG_30` — Pearson r=+0.236, p=1.3e-66
- `DIFF_TOV_RATE_30` — Pearson r=−0.175, p=5.4e-37
- `DIFF_OREB_RATE_30` — Pearson r=+0.057, p=4.7e-05
- `DIFF_FT_RATE_30` — Pearson r=+0.058, p=3.2e-05

**Parameter justification (rolling window sweep):**
- ROLLING_LONG=40 beat 30 on Spread MAE (10.848 vs. 10.862). Longer window gives a more stable baseline, reduces game-to-game noise.
- ROLLING_SHORT=7 beat 10 on Spread MAE (10.856 vs. 10.862). Tighter window picks up injury impacts and hot/cold streaks faster.

**Why:** These are the foundational efficiency metrics in basketball analytics. The differential form implicitly normalizes for opponent strength and reduces multicollinearity compared to including raw stats for both teams separately.


### 3C. Defensive Four Factors Differentials (rolling 40-game averages)

The defensive versions measure what opponents do *against* each team. For each team-game, the opponent's four factors in that game are captured as that team's defensive stat and rolled forward.

**Formula (defensive EFG as example):**
```
Home_DEF_EFG_in_game = Away_EFG_in_game  (what the away team shot against the home defense)
Home_ROLL_DEF_EFG_30 = rolling 40-game average of Home_DEF_EFG
```

**Features used (4 defensive differentials):**
- `DIFF_DEF_EFG_30` — Pearson r=−0.244, p=5.8e-71
- `DIFF_DEF_TOV_RATE_30` — Pearson r=+0.085, p=1.1e-09
- `DIFF_DEF_OREB_RATE_30` — Pearson r=−0.151, p=1.3e-27
- `DIFF_DEF_FT_RATE_30` — Pearson r=−0.091, p=5.2e-11

**Why:** Offensive Four Factors capture what a team does; defensive Four Factors capture what they allow. A team can have elite offense but porous defense — using only offensive differentials misses this. The combined 8-factor framework (4 offensive + 4 defensive diffs) is the full Oliver model.


### 3D. Pace and Expected Possessions

**Formula:**
```
Expected_Pace = (Home_Pace_30 + Away_Pace_30) / 2
Pace_Diff     = Home_Pace_30 - Away_Pace_30
```

Where `Poss ≈ FGA - OREB + TOV + 0.44 * FTA` and `Pace = per-game average of both teams' possession estimates`.

**Features used:**
- `EXPECTED_PACE` — Pearson r with Total: +0.292, p=6.0e-102
- `PACE_DIFF` was dropped from Total model (r=+0.010, p=0.477 — not statistically significant)

**Why:** Pace directly determines the number of scoring opportunities in a game. A game between two fast teams generates 10-15 more possessions than a game between two slow teams. Critical for Total prediction. Pace also affects OREB by increasing the number of missed shots available for offensive rebounds.


### 3E. Net Rating and Offensive/Defensive Rating

**Formula:**
```
ORtg = (PTS / Poss) * 100
DRtg = (OPP_PTS / Poss) * 100
NetRtg = ORtg - DRtg
```

All computed as rolling 40-game (baseline) and 7-game (recent) averages.

**Features used:**
- `DIFF_NET_RTG_30` — Pearson r=+0.370, p=1.4e-167 (second-strongest Spread predictor)
- `DIFF_NET_RTG_10` — Pearson r=+0.342, p=8.5e-142 (recent form version)
- `SUM_ORTG_30` — Pearson r with Total: +0.183, p=3.3e-40
- `SUM_DRTG_30` — Pearson r with Total: +0.228, p=5.0e-62

**Why:** Net rating per 100 possessions is the single best team-level predictor of future wins (r > 0.95 with season win totals). It removes pace effects, isolating pure efficiency. The combined ORtg/DRtg sums directly feed Total prediction.


### 3F. Rest and Schedule Features

**Features used:**
- `HOME_REST_DAYS`, `AWAY_REST_DAYS` (days since last game, capped at 4)
- `HOME_IS_B2B` — Pearson r=−0.059, p=2.4e-05
- `AWAY_IS_B2B` — Pearson r=+0.050, p=3.3e-04
- `REST_DIFF` — Pearson r=+0.064, p=3.9e-06

**Why:** Back-to-back games cost roughly 1.25-2.26 points per published research. This effect is larger on the road. The rest differential captures asymmetric fatigue between teams. Only zero days of rest is statistically significant; 1 day vs. 3+ days shows negligible difference, which is why we cap at 4.

**Note on Total model:** `SUM_B2B` was dropped from Total features — r=−0.005, p=0.748, not statistically significant. B2B fatigue affects who wins (Spread) more than how many points are scored (Total).


### 3G. Recent Form (7-game rolling averages)

We compute 7-game rolling averages of points scored, points allowed, and net rating for both teams.

**Features used:**
- `DIFF_PTS_10` = home_pts_7 − away_pts_7 — Pearson r=+0.196, p=4.8e-46
- `DIFF_NET_RTG_10` = home_net_rtg_7 − away_net_rtg_7 — Pearson r=+0.342, p=8.5e-142
- `SUM_PTS_10` = home_pts_7 + away_pts_7 — Pearson r with Total: +0.310, p=7.5e-116

**Why:** The 7-game window captures hot/cold streaks and recent lineup changes that the 40-game window smooths over. Using both windows simultaneously gives the model access to baseline quality (40-game) and current momentum (7-game). Window reduced from 10 to 7 based on justify_params.py sweep results.


### 3H. Expected Missed Shots (OREB-specific)

**Formula:**
```
Home_Expected_Misses_30 = Home_FGA_30 * (1 - Home_FG%_30)
Away_Expected_Misses_30 = Away_FGA_30 * (1 - Away_FG%_30)
SUM_EXPECTED_MISSES_30  = Home_Expected_Misses_30 + Away_Expected_Misses_30
```

**Pearson r with OREB_TOTAL: +0.212, p=8.6e-54**

**Why:** Offensive rebounds require missed shots. A game with more field goal attempts and/or lower shooting percentages creates more rebound opportunities. This feature directly models the supply side of the OREB equation.


### 3I. OREB% vs. Opponent DREB% Matchup

**Formula:**
```
OREB_matchup_home  = Home_OREB%_30 * (1 - Away_DREB%_30)
OREB_matchup_away  = Away_OREB%_30 * (1 - Home_DREB%_30)
OREB_MATCHUP_TOTAL = OREB_matchup_home + OREB_matchup_away
```

**Feature used:** `OREB_MATCHUP_TOTAL` — Pearson r=+0.291, p=1.3e-101 (strongest OREB predictor)

**Dropped:** `OREB_MATCHUP_DIFF` — r=−0.012, p=0.386. Not statistically significant; the total already captures the interaction. The differential adds no additional signal.

**Why:** A team's offensive rebound rate measures its aggressiveness on the glass, but the opponent's defensive rebound rate determines how many of those opportunities convert. The interaction captures the matchup-specific probability.


### 3J. Win Percentage Differentials

**Features used:**
- `DIFF_WIN_PCT` — Pearson r=+0.254, p=9.9e-77
- `DIFF_HOME_ROAD_WIN_PCT` — Pearson r=+0.232, p=4.4e-64

**Implementation:** Home/road win percentages are tracked separately for each team using `shift(1)` expanding sums on home-only and road-only game subsets. NaN until the team has played at least one game in that venue context.

**Why:** Win percentage is a simple proxy for team quality. The venue-specific differential captures teams that are disproportionately stronger or weaker in specific contexts (e.g., Denver's altitude advantage at home). This adds information beyond net rating because it incorporates clutch performance and closing ability.


### 3K. Roster Availability Score (player-level aggregation)

**Formula:**
```
rolling_prior_min_i = 15-game rolling average of player i's minutes, shifted by 1 game
Team_Availability   = sum(prior_BPM_i × rolling_prior_min_i)  for all players i on the roster
AVAILABILITY_DIFF   = Home_Availability - Away_Availability
```

For historical games, each player's minute contribution is their **15-game rolling average of prior games** (not the current game's actual minutes). This prevents look-ahead bias. For future predictions, the rolling average is replaced by each player's season-average minutes from `LeagueDashPlayerStats`, with injured/doubtful players zeroed out.

**BPM source:** Basketball Reference advanced stats table, loaded from local CSV files `data/bbref_advanced_{year}.csv`. Files required: 2022, 2023, 2024, 2025, 2026. Traded players use their "TOT" (full-season) BPM. Players not found in BBRef receive BPM = 0 (replacement-level assumption).

**Leakage fix — prior-season BPM:** BPM is shifted forward one season: a game in 2024-25 uses 2023-24 BPM, and a game in 2025-26 uses 2024-25 BPM.

**Pearson r with Spread: significant when available** (NaN rows excluded by dropna).


## 4. Feature Sets by Target Variable

### Spread Model Features (17 features)
| Feature | Pearson r | p-value | Rationale |
|---------|-----------|---------|-----------|
| `ELO_DIFF` | +0.375 | 5.1e-172 | Cumulative team strength differential |
| `DIFF_NET_RTG_30` | +0.370 | 1.4e-167 | Overall efficiency differential (40-game) |
| `DIFF_NET_RTG_10` | +0.342 | 8.5e-142 | Recent form efficiency differential (7-game) |
| `DIFF_WIN_PCT` | +0.254 | 9.9e-77 | Season record differential |
| `DIFF_DEF_EFG_30` | −0.244 | 5.8e-71 | Defensive shooting efficiency matchup |
| `DIFF_EFG_30` | +0.236 | 1.3e-66 | Offensive shooting efficiency matchup |
| `DIFF_HOME_ROAD_WIN_PCT` | +0.232 | 4.4e-64 | Venue-specific record differential |
| `DIFF_PTS_10` | +0.196 | 4.8e-46 | Recent scoring differential (7-game) |
| `DIFF_TOV_RATE_30` | −0.175 | 5.4e-37 | Ball security differential |
| `DIFF_DEF_OREB_RATE_30` | −0.151 | 1.3e-27 | Defensive rebounding matchup |
| `DIFF_DEF_FT_RATE_30` | −0.091 | 5.2e-11 | Defensive fouling tendency |
| `DIFF_DEF_TOV_RATE_30` | +0.085 | 1.1e-09 | Forced turnover rate differential |
| `REST_DIFF` | +0.064 | 3.9e-06 | Fatigue asymmetry |
| `HOME_IS_B2B` | −0.059 | 2.4e-05 | Home back-to-back penalty |
| `DIFF_FT_RATE_30` | +0.058 | 3.2e-05 | Free throw generation differential |
| `DIFF_OREB_RATE_30` | +0.057 | 4.7e-05 | Offensive rebounding matchup |
| `AWAY_IS_B2B` | +0.050 | 3.3e-04 | Away back-to-back (benefits home team) |
| `AVAILABILITY_DIFF` | — | — | Roster health (BPM × minutes) |

All 17 statistical features have p < 0.001. AVAILABILITY_DIFF excluded from Pearson analysis due to NaN rows.

### Total Model Features (8 features)
| Feature | Pearson r | p-value | Rationale |
|---------|-----------|---------|-----------|
| `SUM_PTS_10` | +0.310 | 7.5e-116 | Combined recent scoring (7-game) |
| `SUM_PTS_30` | +0.298 | 1.2e-106 | Combined baseline scoring (40-game) |
| `EXPECTED_PACE` | +0.292 | 6.0e-102 | Possessions = scoring opportunities |
| `PACE_X_EFG` | +0.282 | 4.7e-95 | Interaction: pace × combined shooting |
| `SUM_DRTG_30` | +0.228 | 5.0e-62 | Combined defensive quality (leaky = more points) |
| `SUM_ORTG_30` | +0.183 | 3.3e-40 | Combined offensive firepower |
| `SUM_EFG_30` | +0.179 | 1.4e-38 | Combined shooting efficiency |
| `SUM_FT_RATE_30` | +0.097 | 3.0e-12 | Free throw volume adds points |

**Dropped from Total:** `ELO_DIFF` (r=−0.015, p=0.275), `PACE_DIFF` (r=+0.010, p=0.477), `SUM_B2B` (r=−0.005, p=0.748) — all not statistically significant.

### OREB Model Features (10 features)
| Feature | Pearson r | p-value | Rationale |
|---------|-----------|---------|-----------|
| `OREB_MATCHUP_TOTAL` | +0.291 | 1.3e-101 | OREB% vs. opponent DREB% interaction |
| `SUM_OREB_10` | +0.278 | 3.4e-92 | Combined recent OREB form (7-game) |
| `SUM_OREB_RATE_30` | +0.276 | 2.6e-91 | Combined baseline OREB rate (40-game) |
| `SUM_FGA_30` | +0.248 | 1.7e-73 | Shot volume = rebound opportunities |
| `SUM_EXPECTED_MISSES_30` | +0.212 | 8.6e-54 | Supply of rebound chances |
| `AWAY_ROLL_OREB_RATE_30` | +0.206 | 1.4e-50 | Away team individual OREB rate |
| `HOME_ROLL_OREB_RATE_30` | +0.202 | 1.5e-48 | Home team individual OREB rate |
| `HOME_ROLL_DREB_RATE_30` | −0.142 | 1.0e-24 | Home defensive rebounding (denies away OREB) |
| `EXPECTED_PACE` | +0.125 | 2.0e-19 | More possessions = more rebound chances |
| `AWAY_ROLL_DREB_RATE_30` | −0.124 | 3.7e-19 | Away defensive rebounding (denies home OREB) |

**Dropped from OREB:** `OREB_MATCHUP_DIFF` (r=−0.012, p=0.386) — not statistically significant.


## 5. Modeling Strategy

### Spread
**Baseline:** Ridge Regression (alpha=1.0, features standardized).
**Primary models:** XGBoost (`reg:squarederror`) and LightGBM, each validated via walk-forward TimeSeriesSplit.
**Tuning:** Early stopping on a temporal 15% holdout. `n_estimators=1000` with `early_stopping_rounds=50`. Fixed hyperparameters: `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`. Optuna removed — Bayesian search on small dataset overfitted to fold boundaries.
**Ensemble:** Stack Ridge + XGBoost + LightGBM using a Ridge meta-learner (alpha=0.5). Out-of-fold (OOF) predictions from 5-fold KFold train the meta-learner.
**Best-model selection:** After training, all individual model CV MAEs and the ensemble OOF MAE are compared. Predictions route through the single best-performing model automatically.

### Total
**Baseline:** Ridge Regression on raw features.
**Primary model:** LightGBM (`regression` objective). Interaction feature `PACE_X_EFG = expected_pace × SUM_EFG_30` included as explicit input.
**Tuning:** Same early stopping approach as Spread.
**Ensemble + best-model selection:** Same stacking and auto-selection logic as Spread.

### OREB
**Baseline:** Negative Binomial GLM (statsmodels; handles overdispersion in count data).
**Primary model:** XGBoost with `count:poisson` objective.
**Ensemble:** Stack **Negative Binomial GLM** + XGBoost (Poisson) + LightGBM using a Ridge meta-learner. The NB GLM is the first base model to preserve count-data modeling benefit throughout the stack.
**Best-model selection:** Same auto-selection logic; NB GLM CV MAE compared against XGBoost, LightGBM, and ensemble.

### Sample Weights
All models receive recency-based sample weights via `compute_sample_weights()`. Games are weighted by season using exponential decay (`decay_per_season=0.5`):
- 2025-26 season → **1.00**
- 2024-25 season → **0.50**
- 2023-24 season → **0.25**
- 2022-23 season → **0.125**
- 2021-22 season → **0.0625**

Weights are sliced per fold in walk-forward validation and passed through to all base models and the meta-learner.

### Validation
Walk-forward TimeSeriesSplit (5 folds). Report MAE for each target on each fold and the mean ± std. After training, a **MODEL PERFORMANCE SUMMARY** is printed to terminal showing all model MAEs with `*` marking the auto-selected model for predictions.

### Prediction Generation
For each game in the prediction window:
1. Fetch current injury report from ESPN API.
2. Look up each team's most recent rolling features from `team_df`.
3. Compute availability scores using prior-season BPM × season-average minutes, with injured players zeroed out.
4. Compute rest days dynamically: `team_last_game_date` seeded from the last training game per team, updated chronologically through the prediction window for correct B2B detection.
5. Compute the same differential/sum features as training.
6. Feed into the auto-selected best model (Ridge, XGBoost, LightGBM, or Ensemble).
7. Output numeric predictions for Spread, Total, and OREB in `Predictions_filled.csv`.


## 6. Implementation Notes

### Rolling Feature Pipeline
All rolling averages use `shift(1)` before `.rolling()` to ensure only data from prior games is used. `min_periods=5` prevents noisy averages from small early-season samples. Both the 40-game baseline (`_30` columns) and 7-game recent-form (`_10` columns) windows are computed in a single pass over `team_df`. Column suffix labels (`_30`, `_10`) are fixed strings; the actual window sizes are controlled by `ROLLING_LONG` and `ROLLING_SHORT` constants in `build_dataset.py`.

### Availability Score — Training vs. Prediction
- **Training:** Each player's contribution uses a **15-game rolling average of prior minutes** (`shift(1)` then `.rolling(15, min_periods=1).mean()`). BPM taken from the **prior season** to avoid end-of-season leakage. `GAME_DATE` consumed directly from `PlayerGameLogs` to prevent column conflict.
- **Prediction:** `avg_minutes` from `LeagueDashPlayerStats` (latest season), with `MIN_FLOAT = 0` for players in the ESPN injury report. BPM uses the **second-most-recent season** to match training approach.

### Graceful Degradation
All three external data pulls (BBRef BPM, ESPN injuries, `PlayerGameLogs`) are wrapped in `try/except`. If any fails, `AVAILABILITY_DIFF` is set to NaN for affected rows, excluded by `dropna`. The model trains on a slightly smaller dataset but does not crash.

### Column Name Normalization
`nba_api` column casing varies across library versions. All DataFrames returned from nba_api calls are normalized with `.rename(columns=str.upper)` immediately after fetching.


## 7. Parameter Justification Tool (`justify_params.py`)

A standalone script that loads `data/games_raw.csv` (no API calls) and runs two analyses:

**Part 1 — Feature Justification:** For each feature in each model, computes Pearson r + two-tailed t-test p-value, standardized Ridge coefficient, and permutation importance (MAE change when feature is shuffled). Features sorted by |r|. Verdict thresholds: |r| < 0.05 and p > 0.10 → "WEAK - consider dropping".

**Part 2 — Hyperparameter Sweep:** Coordinate descent over 6 parameters (ELO_K, ELO_CARRYOVER, ELO_HOME_BONUS, ROLLING_LONG, ROLLING_SHORT, N_LATE). Each parameter swept across 5-7 values while holding others fixed. Ridge 5-fold CV MAE reported for Spread, Total, and OREB per value. Summary table shows optimal vs. current value for each parameter.

**Usage:**
```
python build_dataset.py   # fetch fresh data first (API calls)
python justify_params.py  # runs offline on saved data
python justify_params.py > results.txt  # save output to file
```

All parameter values in `build_dataset.py` reflect the optimal values found by this sweep as of the last run.
