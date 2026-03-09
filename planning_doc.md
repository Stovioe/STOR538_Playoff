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
Scraped from `basketball-reference.com/leagues/NBA_{year}_advanced.html`. Provides Box Plus/Minus (BPM) for every player in a season. Used as the quality weight in the player availability feature. 3-second delay between requests to respect BBRef rate limits. Traded players are de-duplicated by keeping the "TOT" (season totals) row.

**Not implemented (future work):**
- ESPN schedule API (`scoreboard?dates=YYYYMMDD`) — schedule pre-populated in `Predictions.csv`
- NBA.com Hustle Stats (`LeagueDashPTStats`) — contested rebounds, loose balls; relevant for OREB but deprioritized


## 2. Data Cleaning Summary

**Step 1: Pull raw game logs.** Use `LeagueGameFinder` for the 2023-24, 2024-25, and 2025-26 seasons. Each row is one team's performance in one game. Filter to regular season only.

**Step 2: Reshape to game-level rows.** Each game appears twice in raw data (once per team). Merge on `GAME_ID` to create a single row per game with `HOME_` and `AWAY_` prefixed columns. Identify home/away by checking `MATCHUP` for "vs." (home) or "@" (away).

**Step 3: Compute target variables.**
- `Spread = HOME_PTS - AWAY_PTS`
- `Total = HOME_PTS + AWAY_PTS`
- `OREB = HOME_OREB + AWAY_OREB`

**Step 3b: Player availability scores.** Pull player BPM from BBRef and per-game minutes from `PlayerGameLogs`. For each historical game compute `sum(BPM × rolling_prior_minutes)` per team → `HOME_AVAILABILITY` and `AWAY_AVAILABILITY`. `GAME_DATE` is provided directly by `PlayerGameLogs` and is not re-merged from the game log to avoid column conflicts. Each player's minute contribution uses a 15-game rolling average of prior games (`shift(1)` before rolling) rather than actual game minutes, eliminating look-ahead bias. BPM is taken from the **prior season** (season label shifted forward by one year) to prevent end-of-season leakage: e.g., for a 2024-25 game the model uses 2023-24 BPM. Merged onto the game-level DataFrame before feature engineering.

**Step 4: Handle missing data.** Games with missing box score stats (postponed, incomplete) are dropped. Rolling features require `min_periods=5` — a team's first 4 games of any window are set to NaN and excluded from training via `dropna`.

**Step 5: Sort chronologically.** All games sorted by `GAME_DATE` ascending. Rolling features computed strictly on prior games using `shift(1)` to prevent look-ahead bias.

**Step 6: Train/test split.** Walk-forward validation: train on 2023-24 and 2024-25 seasons, test on 2025-26 through March 13. For final predictions (March 14 through April 12), train on all available data through March 13.


## 3. Engineered Variables

### 3A. Elo Rating (per team, updated after each game)

**Formula:**
```
Elo_new = Elo_old + K * (S - E)
where:
  K = 20
  S = 1 if win, 0 if loss
  E = 1 / (1 + 10^((Elo_opp - Elo_self) / 400))
```

Seasonal carryover: `Elo_start = 0.75 * Elo_end_prev + 0.25 * 1500`

**Feature used:** `ELO_DIFF = Home_Elo - Away_Elo + 100` (the +100 is the home court Elo bonus, roughly 3.5 points).

**Why:** Elo ratings encode cumulative team strength in a single number. FiveThirtyEight demonstrated that standalone Elo achieves 65-68% win prediction accuracy with zero box-score data. The Elo differential is the single best low-dimensional predictor of spread. Cited: Nate Silver, "How We Calculate NBA Elo Ratings," FiveThirtyEight.


### 3B. Offensive Four Factors Differentials (rolling 20-game averages)

Dean Oliver's Four Factors capture 96% of variance in team wins (as measured in the 2024-25 season). We compute each factor as a rolling 20-game average for both teams, then take the differential (home minus away).

| Factor | Formula | Oliver's Weight |
|--------|---------|-----------------|
| Effective FG% | `(FGM + 0.5 * FG3M) / FGA` | 40% |
| Turnover Rate | `TOV / (FGA + 0.44 * FTA + TOV)` | 25% |
| Offensive Rebound Rate | `OREB / (OREB + OPP_DREB)` | 20% |
| Free Throw Rate | `FTM / FGA` | 15% |

**Features used (4 offensive differentials):**
- `DIFF_EFG_20 = home_eFG_20 - away_eFG_20`
- `DIFF_TOV_RATE_20 = home_TOV_rate_20 - away_TOV_rate_20`
- `DIFF_OREB_RATE_20 = home_OREB_rate_20 - away_OREB_rate_20`
- `DIFF_FT_RATE_20 = home_FT_rate_20 - away_FT_rate_20`

**Why:** These are the foundational efficiency metrics in basketball analytics. The differential form implicitly normalizes for opponent strength and reduces multicollinearity compared to including raw stats for both teams separately. The 20-game window balances stability and responsiveness; research shows it slightly outperforms 10-game windows.


### 3C. Defensive Four Factors Differentials (rolling 20-game averages)

The defensive versions measure what opponents do *against* each team. For each team-game, the opponent's four factors in that game are captured as that team's defensive stat and rolled forward.

**Formula (defensive EFG as example):**
```
Home_DEF_EFG_in_game = Away_EFG_in_game  (what the away team shot against the home defense)
Home_ROLL_DEF_EFG_20 = rolling 20-game average of Home_DEF_EFG
```

**Features used (4 defensive differentials):**
- `DIFF_DEF_EFG_20 = home_DEF_EFG_20 - away_DEF_EFG_20` (negative = home allows more efficient shots = weaker defense)
- `DIFF_DEF_TOV_RATE_20` — forced turnover rate differential
- `DIFF_DEF_OREB_RATE_20` — opponent offensive rebound rate allowed
- `DIFF_DEF_FT_RATE_20` — opponent free throw rate allowed

**Why:** Offensive Four Factors capture what a team does; defensive Four Factors capture what they allow. A team can have elite offense but porous defense — using only offensive differentials misses this. The combined 8-factor framework (4 offensive + 4 defensive diffs) is the full Oliver model. The model learns the sign direction from data, so no manual direction-correction is needed.


### 3D. Pace and Expected Possessions

**Formula:**
```
Expected_Pace = (Home_Pace_20 + Away_Pace_20) / 2
```

Where `Poss ≈ FGA - OREB + TOV + 0.44 * FTA` and `Pace = per-game average of both teams' possession estimates`.

**Features used:**
- `EXPECTED_PACE` (average of both teams' 20-game rolling pace)
- `PACE_DIFF` (home pace minus away pace)

**Why:** Pace directly determines the number of scoring opportunities in a game. A game between two fast teams (e.g., Sacramento and Indiana) generates 10-15 more possessions than a game between two slow teams (e.g., Cleveland and New York). This is critical for Total prediction. Pace also affects OREB by increasing the number of missed shots available for offensive rebounds.


### 3E. Net Rating and Offensive/Defensive Rating

**Formula:**
```
ORtg = (PTS / Poss) * 100
DRtg = (OPP_PTS / Poss) * 100
NetRtg = ORtg - DRtg
```

All computed as rolling 20-game averages.

**Features used:**
- `DIFF_NET_RTG_20 = home_net_rtg_20 - away_net_rtg_20`
- `SUM_ORTG_20 = home_ORtg_20 + away_ORtg_20` (combined offensive output, relevant for Total)
- `SUM_DRTG_20 = home_DRtg_20 + away_DRtg_20` (combined defensive quality, relevant for Total)

**Why:** Net rating per 100 possessions is the single best team-level predictor of future wins (r > 0.95 with season win totals). It removes pace effects, isolating pure efficiency. The combined ORtg sum directly feeds Total prediction.


### 3F. Rest and Schedule Features

**Features used:**
- `HOME_REST_DAYS`, `AWAY_REST_DAYS` (days since last game, capped at 4)
- `HOME_IS_B2B`, `AWAY_IS_B2B` (binary: 1 if back-to-back, 0 otherwise)
- `REST_DIFF = HOME_REST_DAYS - AWAY_REST_DAYS`

**Why:** Back-to-back games cost roughly 1.25-2.26 points per published research. This effect is larger on the road. The rest differential captures asymmetric fatigue between teams. Only zero days of rest is statistically significant; 1 day vs. 3+ days shows negligible difference, which is why we cap at 4.


### 3G. Recent Form (5-game rolling averages)

We compute 5-game rolling averages of points scored, points allowed, and net rating for both teams.

**Features used:**
- `DIFF_PTS_5 = home_pts_5 - away_pts_5` (recent scoring differential)
- `DIFF_NET_RTG_5 = home_net_rtg_5 - away_net_rtg_5` (recent efficiency differential)
- `SUM_OREB_5 = home_OREB_5 + away_OREB_5` (recent offensive rebounding form)

**Why:** The 5-game window captures hot/cold streaks and recent lineup changes that the 20-game window smooths over. Using both windows simultaneously gives the model access to baseline quality (20-game) and current momentum (5-game).


### 3H. Expected Missed Shots (OREB-specific)

**Formula:**
```
Home_Expected_Misses_20 = Home_FGA_20 * (1 - Home_FG%_20)
Away_Expected_Misses_20 = Away_FGA_20 * (1 - Away_FG%_20)
Total_Expected_Misses = Home_Expected_Misses_20 + Away_Expected_Misses_20
```

**Features used:**
- `SUM_EXPECTED_MISSES_20` (sum of both teams' rolling expected misses)

**Why:** Offensive rebounds require missed shots. A game with more field goal attempts and/or lower shooting percentages creates more rebound opportunities. This feature directly models the supply side of the OREB equation. Teams that shoot poorly but take many shots (high-pace, low-efficiency) generate more OREB chances.


### 3I. OREB% vs. Opponent DREB% Matchup

**Formula:**
```
OREB_matchup_home = Home_OREB%_20 * (1 - Away_DREB%_20)
OREB_matchup_away = Away_OREB%_20 * (1 - Home_DREB%_20)
Total_OREB_matchup = OREB_matchup_home + OREB_matchup_away
```

**Features used:**
- `OREB_MATCHUP_TOTAL`
- `OREB_MATCHUP_DIFF = OREB_matchup_home - OREB_matchup_away`

**Why:** A team's offensive rebound rate measures its aggressiveness on the glass, but the opponent's defensive rebound rate determines how many of those opportunities convert. The interaction captures the matchup-specific probability. A high-OREB% team facing a low-DREB% team will generate more offensive rebounds than either stat alone would predict.


### 3J. Win Percentage Differentials

**Features used:**
- `DIFF_WIN_PCT = home_win_pct - away_win_pct` (season-to-date overall)
- `DIFF_HOME_ROAD_WIN_PCT = home_HOME_win_pct - away_ROAD_win_pct` (venue-specific records)

**Implementation:** Home/road win percentages are tracked separately for each team using `shift(1)` expanding sums on home-only and road-only game subsets. NaN until the team has played at least one game in that venue context.

**Why:** Win percentage is a simple proxy for team quality. The venue-specific differential captures teams that are disproportionately stronger or weaker in specific contexts (e.g., Denver's altitude advantage at home, teams with strong away records from a road-heavy schedule early in the season). This adds information beyond net rating because it incorporates clutch performance and closing ability.


### 3K. Roster Availability Score (player-level aggregation)

**Formula:**
```
rolling_prior_min_i = 15-game rolling average of player i's minutes, shifted by 1 game
Team_Availability   = sum(prior_BPM_i × rolling_prior_min_i)  for all players i on the roster
AVAILABILITY_DIFF   = Home_Availability - Away_Availability
```

For historical games, each player's minute contribution is their **15-game rolling average of prior games** (not the current game's actual minutes). This prevents look-ahead bias — we only know who played and for how long *after* the game ends. For future predictions, the rolling average is replaced by each player's season-average minutes from `LeagueDashPlayerStats`, with injured/doubtful players zeroed out.

**BPM source:** Basketball Reference advanced stats table (`NBA_{year}_advanced.html`), loaded from local CSV files saved as `data/bbref_advanced_{year}.csv`. Traded players use their "TOT" (full-season) BPM. Players not found in BBRef receive BPM = 0 (replacement-level assumption).

**Leakage fix — prior-season BPM:** BPM is a season-long metric that is only finalized at the end of the season. Using the current season's BPM for games early in that season would incorporate future information. To prevent this, BPM is **shifted forward one season**: a game in 2024-25 uses 2023-24 BPM, and a game in 2025-26 uses 2024-25 BPM. This applies consistently in both training (`compute_team_availability_scores`) and prediction (`compute_prediction_availability`).

**Why this design solves the roster-turnover problem:** The availability score updates automatically when a player is traded (their TEAM_ID changes in the stats table) or injured (their minute contribution is zeroed out from the ESPN injury report). A team missing its top two players by BPM × minutes might be missing 60–80 availability points — a signal that no team-level rolling average can capture until several games have been played with the new lineup.

**Why BPM × rolling minutes rather than raw BPM:** BPM alone ignores usage. A star player's injury costs more than a bench player's injury. Multiplying by rolling prior minutes weights each player's contribution by how much they typically play, making the aggregate a proper measure of total quality on the floor without peeking at the current game.


## 4. Feature Sets by Target Variable

### Spread Model Features
| Feature | Rationale |
|---------|-----------|
| `ELO_DIFF` | Baseline cumulative team strength differential |
| `DIFF_EFG_20` | Primary shooting efficiency matchup |
| `DIFF_TOV_RATE_20` | Ball security differential |
| `DIFF_OREB_RATE_20` | Offensive rebounding matchup |
| `DIFF_FT_RATE_20` | Free throw generation differential |
| `DIFF_DEF_EFG_20` | Defensive shooting efficiency matchup |
| `DIFF_DEF_TOV_RATE_20` | Forced turnover rate differential |
| `DIFF_DEF_OREB_RATE_20` | Defensive rebounding matchup |
| `DIFF_DEF_FT_RATE_20` | Defensive free throw rate matchup |
| `DIFF_NET_RTG_20` | Overall efficiency differential |
| `DIFF_NET_RTG_5` | Recent form differential |
| `DIFF_PTS_5` | Recent scoring differential |
| `DIFF_WIN_PCT` | Season record differential |
| `DIFF_HOME_ROAD_WIN_PCT` | Venue-specific record differential |
| `REST_DIFF` | Fatigue asymmetry |
| `HOME_IS_B2B`, `AWAY_IS_B2B` | Back-to-back penalties |
| `AVAILABILITY_DIFF` | Roster health differential (BPM × minutes) |

### Total Model Features
| Feature | Rationale |
|---------|-----------|
| `EXPECTED_PACE` | Determines possession count |
| `PACE_DIFF` | Pace mismatch indicator |
| `SUM_ORTG_20` | Combined offensive firepower |
| `SUM_DRTG_20` | Combined defensive quality |
| `SUM_EFG_20` | Combined shooting efficiency |
| `SUM_FT_RATE_20` | Combined free throw generation |
| `SUM_PTS_20` | Combined recent scoring (20-game) |
| `SUM_PTS_5` | Recent combined scoring (5-game) |
| `PACE_X_EFG` | Interaction: possessions × shooting efficiency |
| `SUM_B2B` | Total fatigue in game |
| `ELO_DIFF` | Blowout-prone matchups affect total |

### OREB Model Features
| Feature | Rationale |
|---------|-----------|
| `SUM_OREB_RATE_20` | Combined board-crashing tendency |
| `OREB_MATCHUP_TOTAL` | Interaction of OREB% vs. opponent DREB% |
| `OREB_MATCHUP_DIFF` | Matchup asymmetry |
| `SUM_EXPECTED_MISSES_20` | Supply of rebound opportunities |
| `EXPECTED_PACE` | Volume driver for all counting stats |
| `SUM_OREB_5` | Recent OREB form |
| `SUM_FGA_20` | Shot volume proxy |
| `HOME_ROLL_OREB_RATE_20`, `AWAY_ROLL_OREB_RATE_20` | Individual team OREB rates |
| `HOME_ROLL_DREB_RATE_20`, `AWAY_ROLL_DREB_RATE_20` | Individual team DREB rates |


## 5. Modeling Strategy

### Spread
**Baseline:** Ridge Regression (alpha=1.0, features standardized).
**Primary models:** XGBoost (`reg:squarederror`) and LightGBM, each validated via walk-forward TimeSeriesSplit.
**Tuning:** Early stopping on a temporal 15% holdout (last 15% of training rows by date). `n_estimators=1000` with `early_stopping_rounds=50` for both XGBoost and LightGBM. Fixed hyperparameters: `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`. Optuna was removed — Bayesian search over 3-fold inner CV on a small dataset overfitted to specific fold boundaries and consistently worsened held-out MAE.
**Ensemble:** Stack Ridge + XGBoost + LightGBM using a Ridge meta-learner (alpha=0.5). Out-of-fold (OOF) predictions from 5-fold KFold used to train the meta-learner. Ensemble OOF MAE is printed alongside Ridge CV MAE for direct comparison.

### Total
**Baseline:** Ridge Regression on raw features.
**Primary model:** LightGBM (`regression` objective). Interaction feature `PACE_X_EFG = expected_pace × SUM_EFG_20` included as explicit input.
**Tuning:** Same early stopping approach as Spread (15% temporal holdout, `early_stopping_rounds=50`). No Optuna.
**Note:** No log transform needed. At ~220 points per game the distribution is approximately Normal.
**Ensemble:** Stack Ridge + XGBoost + LightGBM using a Ridge meta-learner.

### OREB
**Baseline:** Negative Binomial GLM (statsmodels; handles overdispersion in count data). Research shows NB roughly halves prediction error vs. Poisson for overdispersed NBA counts.
**Primary model:** XGBoost with `count:poisson` objective.
**Ensemble:** Stack **Negative Binomial GLM** + XGBoost (Poisson) + LightGBM using a Ridge meta-learner. The NB GLM is the first base model (not Ridge) to preserve the count-data modeling benefit throughout the stack.
**Note on `predict_nb`:** `sm.add_constant(..., has_constant="add")` used in both training and prediction to prevent double-adding a constant column.

### Sample Weights
All models receive recency-based sample weights via `compute_sample_weights()`. Games are weighted by season:
- 2025-26 season → **1.00**
- 2024-25 season → **0.50**
- 2023-24 season → **0.25**

Weights use exponential decay with `decay_per_season=0.5`. The weights are sliced per fold in walk-forward validation and passed through to all base models and the meta-learner in the stacked ensemble.

### Validation
Walk-forward TimeSeriesSplit (5 folds). Report MAE for each target on each fold and the mean ± std. All candidate models evaluated on the same splits before selecting ensemble weights. After ensemble training, the ensemble's OOF MAE is printed alongside the baseline Ridge CV MAE to diagnose whether stacking provides a meaningful lift.

### Prediction Generation
For each game March 14 through April 12:
1. Fetch current injury report from ESPN API.
2. Look up each team's most recent rolling features from `team_df`.
3. Compute availability scores using prior-season BPM × season-average minutes, with injured players zeroed out.
4. Compute rest days dynamically: `team_last_game_date` is seeded from the last training game per team, then updated after each prediction game is processed in chronological order. This ensures back-to-back detection works correctly throughout the prediction window, not just on the first day after the training cutoff.
5. Compute the same differential/sum features as training via `build_model_features_for_pred`.
6. Feed into the trained stacked ensemble.
7. Output numeric predictions for Spread, Total, and OREB in `Predictions_filled.csv`.


## 6. Implementation Notes

### Rolling Feature Pipeline
All rolling averages use `shift(1)` before `.rolling()` to ensure only data from prior games is used. `min_periods=5` prevents noisy averages from small early-season samples (previously `min_periods=3`). Both 20-game (baseline) and 5-game (recent form) windows are computed in a single pass over `team_df`.

### Availability Score — Training vs. Prediction
- **Training:** Each player's contribution uses a **15-game rolling average of prior minutes** (`shift(1)` then `.rolling(15, min_periods=1).mean()`), not the current game's actual minutes. This prevents look-ahead bias — actual minutes are only known after a game concludes. BPM is taken from the **prior season** to avoid end-of-season leakage (e.g., 2024-25 games use 2023-24 BPM). `GAME_DATE` is consumed directly from `PlayerGameLogs` rather than re-merged from the game log, preventing the `GAME_DATE_x / GAME_DATE_y` column conflict that caused a `KeyError` in earlier versions.
- **Prediction:** `avg_minutes` from `LeagueDashPlayerStats` (latest season), with `MIN_FLOAT = 0` for players in the ESPN injury report. BPM uses the **second-most-recent season** (prior season) to match the training approach. This directly solves roster-turnover: the feature updates immediately when a player is traded (new TEAM_ID) or injured (zeroed out).

### Graceful Degradation
All three external data pulls (BBRef BPM, ESPN injuries, `PlayerGameLogs`) are wrapped in `try/except`. If any fails, `AVAILABILITY_DIFF` is set to NaN for affected rows, which are then excluded by `dropna` in `prepare_train_test`. The model trains on a slightly smaller dataset but does not crash.

### Column Name Normalization
`nba_api` column casing varies across library versions. All DataFrames returned from nba_api calls are normalized with `.rename(columns=str.upper)` immediately after fetching.
