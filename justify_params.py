"""
Parameter Justification Tool
=============================
Sweeps key hyperparameters one at a time (coordinate descent) using the
already-saved games_raw.csv — no API calls required.

For each parameter, it recomputes Elo and/or rolling features and evaluates
Ridge cross-validation MAE on Spread, Total, and OREB targets.

Usage:
    python justify_params.py

Output:
    Terminal table showing MAE for each parameter value, with * marking
    the current setting and BEST marking the optimal value found.
"""

import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = "data/games_raw.csv"
CURRENT_SEASON = "2025-26"

# ── Current (baseline) parameter values ──────────────────────────────────────
DEFAULTS = {
    "ELO_K":         16,
    "ELO_CARRYOVER": 0.5,
    "ELO_HOME_BONUS":150,
    "ROLLING_LONG":  40,
    "ROLLING_SHORT": 7,
    "N_LATE":        10,
}

# ── Search grids ──────────────────────────────────────────────────────────────
GRIDS = {
    "ELO_K":          [16, 20, 24, 28, 32, 40, 48],
    "ELO_CARRYOVER":  [0.50, 0.60, 0.70, 0.75, 0.80, 0.90],
    "ELO_HOME_BONUS": [50, 75, 100, 125, 150],
    "ROLLING_LONG":   [15, 20, 25, 30, 40, 50],
    "ROLLING_SHORT":  [5, 7, 10, 12, 15],
    "N_LATE":         [0, 5, 8, 10, 12, 15],
}

SPREAD_FEATURES = [
    "ELO_DIFF",
    "DIFF_EFG_30", "DIFF_TOV_RATE_30", "DIFF_OREB_RATE_30", "DIFF_FT_RATE_30",
    "DIFF_DEF_EFG_30", "DIFF_DEF_TOV_RATE_30", "DIFF_DEF_OREB_RATE_30", "DIFF_DEF_FT_RATE_30",
    "DIFF_NET_RTG_30", "DIFF_NET_RTG_10", "DIFF_PTS_10",
    "DIFF_WIN_PCT", "DIFF_HOME_ROAD_WIN_PCT",
    "REST_DIFF", "HOME_IS_B2B", "AWAY_IS_B2B",
]

TOTAL_FEATURES = [
    "EXPECTED_PACE",
    "SUM_ORTG_30", "SUM_DRTG_30", "SUM_EFG_30", "SUM_FT_RATE_30",
    "SUM_PTS_30", "SUM_PTS_10",
    "PACE_X_EFG",
]

OREB_FEATURES = [
    "SUM_OREB_RATE_30", "OREB_MATCHUP_TOTAL",
    "SUM_EXPECTED_MISSES_30", "EXPECTED_PACE", "SUM_OREB_10",
    "SUM_FGA_30",
    "HOME_ROLL_OREB_RATE_30", "AWAY_ROLL_OREB_RATE_30",
    "HOME_ROLL_DREB_RATE_30", "AWAY_ROLL_DREB_RATE_30",
]


# ── Core feature engineering (no API, works on games_raw.csv) ─────────────────

def compute_elo(games, elo_k, elo_carryover, elo_home_bonus, elo_init=1500):
    elo = {}
    prev_season = None
    home_elos, away_elos = [], []

    for _, row in games.iterrows():
        season = row["SEASON"]
        h_id = row["HOME_TEAM_ID"]
        a_id = row["AWAY_TEAM_ID"]

        if season != prev_season and prev_season is not None:
            for tid in elo:
                elo[tid] = elo_carryover * elo[tid] + (1 - elo_carryover) * elo_init
        prev_season = season

        elo.setdefault(h_id, elo_init)
        elo.setdefault(a_id, elo_init)

        home_elos.append(elo[h_id])
        away_elos.append(elo[a_id])

        h_elo = elo[h_id] + elo_home_bonus
        a_elo = elo[a_id]
        e_home = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        s_home = 1.0 if row["HOME_PTS"] > row["AWAY_PTS"] else 0.0

        margin = abs(row["HOME_PTS"] - row["AWAY_PTS"])
        k_mult = min(math.log(margin + 1), 2.5)
        elo[h_id] += elo_k * k_mult * (s_home - e_home)
        elo[a_id] += elo_k * k_mult * ((1 - s_home) - (1 - e_home))

    games = games.copy()
    games["HOME_ELO"] = home_elos
    games["AWAY_ELO"] = away_elos
    games["ELO_DIFF"] = games["HOME_ELO"] - games["AWAY_ELO"] + elo_home_bonus
    return games


ROLL_STATS = [
    "PTS", "OREB", "DREB", "FGA", "FGM", "FG3M", "FTM", "FTA", "TOV",
    "EFG", "TOV_RATE", "OREB_RATE", "DREB_RATE", "FT_RATE",
    "ORTG", "DRTG", "NET_RTG", "PACE", "EXPECTED_MISSES",
]
DEF_STATS = ["DEF_EFG", "DEF_TOV_RATE", "DEF_OREB_RATE", "DEF_FT_RATE"]


def compute_rolling(games, rolling_long, rolling_short):
    team_rows = []
    for _, row in games.iterrows():
        gd = row["GAME_DATE"]
        gid = row["GAME_ID"]
        for side, opp in [("HOME", "AWAY"), ("AWAY", "HOME")]:
            rec = {"TEAM_ID": row[f"{side}_TEAM_ID"], "GAME_DATE": gd, "GAME_ID": gid}
            for stat in ROLL_STATS:
                rec[stat] = row.get(f"{side}_{stat}", np.nan)
            rec["PTS_ALLOWED"] = row[f"{opp}_PTS"]
            rec["WIN"] = 1 if row[f"{side}_PTS"] > row[f"{opp}_PTS"] else 0
            for ds in DEF_STATS:
                raw = ds.replace("DEF_", "")
                rec[ds] = row.get(f"{opp}_{raw}", np.nan)
            rec["IS_HOME_GAME"] = 1 if side == "HOME" else 0
            team_rows.append(rec)

    tdf = pd.DataFrame(team_rows).sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    for stat in ROLL_STATS + ["PTS_ALLOWED", "WIN"]:
        for window, suffix in [(rolling_long, "30"), (rolling_short, "10")]:
            tdf[f"ROLL_{stat}_{suffix}"] = (
                tdf.groupby("TEAM_ID")[stat]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=5).mean())
            )

    for stat in DEF_STATS:
        tdf[f"ROLL_{stat}_30"] = (
            tdf.groupby("TEAM_ID")[stat]
            .transform(lambda x: x.shift(1).rolling(rolling_long, min_periods=5).mean())
        )

    tdf["SEASON_WIN_PCT"] = (
        tdf.groupby("TEAM_ID")["WIN"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )

    def _home_win_pct(g):
        sw = (g["WIN"] * g["IS_HOME_GAME"]).shift(1)
        sh = g["IS_HOME_GAME"].shift(1)
        return sw.expanding().sum() / sh.expanding().sum().clip(lower=1e-9)

    def _road_win_pct(g):
        sw = (g["WIN"] * (1 - g["IS_HOME_GAME"])).shift(1)
        sh = (1 - g["IS_HOME_GAME"]).shift(1)
        return sw.expanding().sum() / sh.expanding().sum().clip(lower=1e-9)

    tdf["HOME_WIN_PCT"] = tdf.groupby("TEAM_ID", group_keys=False).apply(_home_win_pct)
    tdf["ROAD_WIN_PCT"] = tdf.groupby("TEAM_ID", group_keys=False).apply(_road_win_pct)

    home_seen = tdf.groupby("TEAM_ID")["IS_HOME_GAME"].transform(lambda x: x.shift(1).expanding().sum())
    road_seen = tdf.groupby("TEAM_ID")["IS_HOME_GAME"].transform(lambda x: (1 - x).shift(1).expanding().sum())
    tdf.loc[home_seen < 1, "HOME_WIN_PCT"] = np.nan
    tdf.loc[road_seen < 1, "ROAD_WIN_PCT"] = np.nan

    tdf["PREV_GAME_DATE"] = tdf.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    tdf["REST_DAYS"] = (tdf["GAME_DATE"] - tdf["PREV_GAME_DATE"]).dt.days.clip(upper=4)
    tdf["IS_B2B"] = (tdf["REST_DAYS"] == 1).astype(int)

    return tdf


def merge_rolling(games, tdf):
    roll_cols = [c for c in tdf.columns if c.startswith("ROLL_")] + [
        "SEASON_WIN_PCT", "HOME_WIN_PCT", "ROAD_WIN_PCT", "REST_DAYS", "IS_B2B",
    ]
    mc = ["GAME_ID", "TEAM_ID"] + roll_cols

    home_m = tdf[mc].rename(columns={c: f"HOME_{c}" for c in roll_cols}).rename(columns={"TEAM_ID": "HOME_TEAM_ID"})
    away_m = tdf[mc].rename(columns={c: f"AWAY_{c}" for c in roll_cols}).rename(columns={"TEAM_ID": "AWAY_TEAM_ID"})

    games = games.merge(home_m, on=["GAME_ID", "HOME_TEAM_ID"], how="left")
    games = games.merge(away_m, on=["GAME_ID", "AWAY_TEAM_ID"], how="left")
    return games


def build_features(games):
    g = games.copy()

    for stat in ["EFG", "TOV_RATE", "OREB_RATE", "FT_RATE", "NET_RTG", "ORTG", "DRTG", "PTS"]:
        g[f"DIFF_{stat}_30"] = g[f"HOME_ROLL_{stat}_30"] - g[f"AWAY_ROLL_{stat}_30"]
    for stat in ["NET_RTG", "PTS"]:
        g[f"DIFF_{stat}_10"] = g[f"HOME_ROLL_{stat}_10"] - g[f"AWAY_ROLL_{stat}_10"]

    g["DIFF_WIN_PCT"] = g["HOME_SEASON_WIN_PCT"] - g["AWAY_SEASON_WIN_PCT"]
    g["REST_DIFF"] = g["HOME_REST_DAYS"] - g["AWAY_REST_DAYS"]

    for stat in DEF_STATS:
        g[f"DIFF_{stat}_30"] = g[f"HOME_ROLL_{stat}_30"] - g[f"AWAY_ROLL_{stat}_30"]

    g["DIFF_HOME_ROAD_WIN_PCT"] = g["HOME_HOME_WIN_PCT"] - g["AWAY_ROAD_WIN_PCT"]

    for stat in ["ORTG", "DRTG", "EFG", "FT_RATE", "PACE", "PTS"]:
        g[f"SUM_{stat}_30"] = g[f"HOME_ROLL_{stat}_30"] + g[f"AWAY_ROLL_{stat}_30"]

    g["SUM_PTS_10"] = g["HOME_ROLL_PTS_10"] + g["AWAY_ROLL_PTS_10"]
    g["EXPECTED_PACE"] = (g["HOME_ROLL_PACE_30"] + g["AWAY_ROLL_PACE_30"]) / 2
    g["PACE_DIFF"] = g["HOME_ROLL_PACE_30"] - g["AWAY_ROLL_PACE_30"]
    g["PACE_X_EFG"] = g["EXPECTED_PACE"] * g["SUM_EFG_30"]
    g["SUM_B2B"] = g["HOME_IS_B2B"] + g["AWAY_IS_B2B"]

    g["SUM_OREB_RATE_30"] = g["HOME_ROLL_OREB_RATE_30"] + g["AWAY_ROLL_OREB_RATE_30"]
    g["SUM_OREB_10"] = g["HOME_ROLL_OREB_10"] + g["AWAY_ROLL_OREB_10"]
    g["SUM_EXPECTED_MISSES_30"] = g["HOME_ROLL_EXPECTED_MISSES_30"] + g["AWAY_ROLL_EXPECTED_MISSES_30"]
    g["SUM_FGA_30"] = g["HOME_ROLL_FGA_30"] + g["AWAY_ROLL_FGA_30"]

    g["OREB_MATCHUP_HOME"] = g["HOME_ROLL_OREB_RATE_30"] * (1 - g["AWAY_ROLL_DREB_RATE_30"])
    g["OREB_MATCHUP_AWAY"] = g["AWAY_ROLL_OREB_RATE_30"] * (1 - g["HOME_ROLL_DREB_RATE_30"])
    g["OREB_MATCHUP_TOTAL"] = g["OREB_MATCHUP_HOME"] + g["OREB_MATCHUP_AWAY"]
    g["OREB_MATCHUP_DIFF"] = g["OREB_MATCHUP_HOME"] - g["OREB_MATCHUP_AWAY"]

    # Pass through individual rolling rates needed by OREB_FEATURES
    for col in ["HOME_ROLL_OREB_RATE_30", "AWAY_ROLL_OREB_RATE_30",
                "HOME_ROLL_DREB_RATE_30", "AWAY_ROLL_DREB_RATE_30"]:
        g[col] = g[col]  # already present

    return g


def filter_late_season(games, n_late, current_season=CURRENT_SEASON):
    if n_late == 0:
        return games
    remove_ids = set()
    for season in games["SEASON"].unique():
        if season == current_season:
            continue
        sg = games[games["SEASON"] == season].copy()
        team_ids = pd.concat([sg["HOME_TEAM_ID"], sg["AWAY_TEAM_ID"]]).unique()
        for tid in team_ids:
            team_games = sg[(sg["HOME_TEAM_ID"] == tid) | (sg["AWAY_TEAM_ID"] == tid)].sort_values("GAME_DATE")
            if len(team_games) > n_late:
                late_ids = set(team_games.tail(n_late)["GAME_ID"])
                remove_ids |= late_ids
    return games[~games["GAME_ID"].isin(remove_ids)].reset_index(drop=True)


def evaluate(games, features, target):
    """Ridge CV MAE (5-fold) on completed games with all features present."""
    cols = features + [target]
    df = games.dropna(subset=cols)
    if len(df) < 50:
        return np.nan
    X = df[features].values
    y = df[target].values
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_absolute_error")
    return -scores.mean()


# ── Build the dataset once per unique (elo_params, rolling_params, n_late) ────

def build_dataset(params):
    elo_k        = params["ELO_K"]
    elo_carryover= params["ELO_CARRYOVER"]
    elo_home_bonus=params["ELO_HOME_BONUS"]
    rolling_long = params["ROLLING_LONG"]
    rolling_short= params["ROLLING_SHORT"]
    n_late       = params["N_LATE"]

    games = pd.read_csv(DATA_PATH, parse_dates=["GAME_DATE"])
    games = games.sort_values("GAME_DATE").reset_index(drop=True)
    games = compute_elo(games, elo_k, elo_carryover, elo_home_bonus)
    tdf = compute_rolling(games, rolling_long, rolling_short)
    games = merge_rolling(games, tdf)
    games = build_features(games)
    games = filter_late_season(games, n_late)
    return games


# ── Feature justification ─────────────────────────────────────────────────────

def justify_features(games, features, target, model_name):
    """
    For each feature print:
      - Pearson r with target + p-value
      - Standardized Ridge coefficient (all features scaled, so magnitudes comparable)
      - Permutation importance (MAE drop when feature is shuffled)
    """
    cols = features + [target]
    df = games.dropna(subset=cols)
    X = df[features].values
    y = df[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Pearson correlations
    corrs = []
    pvals = []
    for i, feat in enumerate(features):
        r, p = scipy_stats.pearsonr(X_scaled[:, i], y)
        corrs.append(r)
        pvals.append(p)

    # Standardized Ridge coefficients
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    coefs = ridge.coef_

    # Permutation importance (MAE reduction when each feature is shuffled)
    perm = permutation_importance(ridge, X_scaled, y, n_repeats=20,
                                  scoring="neg_mean_absolute_error",
                                  random_state=42)
    perm_means = perm.importances_mean  # negative = MAE goes up when shuffled (good feature)

    # Sort by abs(correlation)
    order = sorted(range(len(features)), key=lambda i: abs(corrs[i]), reverse=True)

    model_mae = cross_val_score(
        Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=1.0))]),
        X, y, cv=5, scoring="neg_mean_absolute_error"
    ).mean() * -1

    print(f"\n{'='*80}")
    print(f"  FEATURE JUSTIFICATION: {model_name}  (n={len(df)}, Ridge CV MAE={model_mae:.4f})")
    print(f"{'='*80}")
    print(f"  {'Feature':<28}  {'Pearson r':>9}  {'p-value':>10}  {'Ridge coef':>11}  {'Perm imp':>9}  Verdict")
    print(f"  {'-'*28}  {'-'*9}  {'-'*10}  {'-'*11}  {'-'*9}  -------")

    KEEP_THRESHOLD_R = 0.05      # drop if |r| < 0.05 (nearly no signal)
    KEEP_THRESHOLD_P = 0.10      # warn if p > 0.10 (not statistically significant)

    for i in order:
        feat = features[i]
        r    = corrs[i]
        p    = pvals[i]
        coef = coefs[i]
        pi   = -perm_means[i]    # flip sign: positive = shuffling hurts MAE = feature is useful

        # Verdict
        if abs(r) < KEEP_THRESHOLD_R and p > KEEP_THRESHOLD_P:
            verdict = "WEAK - consider dropping"
        elif p > KEEP_THRESHOLD_P:
            verdict = "not significant (p>{:.2f})".format(KEEP_THRESHOLD_P)
        elif pi < 0:
            verdict = "low perm imp"
        else:
            verdict = "justified"

        sig = "***" if p < 0.001 else ("** " if p < 0.01 else ("*  " if p < 0.05 else "   "))
        print(f"  {feat:<28}  {r:>+9.4f}  {p:>10.2e}{sig}  {coef:>+11.4f}  {pi:>+9.4f}  {verdict}")

    print(f"\n  Significance: *** p<0.001   ** p<0.01   * p<0.05")


# ── Sweep logic ───────────────────────────────────────────────────────────────

def sweep_param(param_name, grid, base_params):
    results = []
    for val in grid:
        params = {**base_params, param_name: val}
        games = build_dataset(params)
        mae_spread = evaluate(games, SPREAD_FEATURES, "SPREAD")
        mae_total  = evaluate(games, TOTAL_FEATURES,  "TOTAL")
        mae_oreb   = evaluate(games, OREB_FEATURES,   "OREB_TOTAL")
        results.append({
            "value":      val,
            "spread_mae": mae_spread,
            "total_mae":  mae_total,
            "oreb_mae":   mae_oreb,
        })
    return results


def print_sweep(param_name, results, default_val):
    best_spread = min(results, key=lambda r: r["spread_mae"])["value"]
    best_total  = min(results, key=lambda r: r["total_mae"])["value"]
    best_oreb   = min(results, key=lambda r: r["oreb_mae"])["value"]

    header = f"\n{'='*65}\n  {param_name}   (current={default_val})\n{'='*65}"
    print(header)
    print(f"  {'Value':>10}  {'Spread MAE':>12}  {'Total MAE':>12}  {'OREB MAE':>10}  Notes")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*10}  -----")
    for r in results:
        v = r["value"]
        notes = []
        if v == default_val:
            notes.append("current")
        if v == best_spread:
            notes.append("best-spread")
        if v == best_total:
            notes.append("best-total")
        if v == best_oreb:
            notes.append("best-oreb")
        flag = " *" if v == default_val else "  "
        print(f"{flag} {str(v):>10}  {r['spread_mae']:>12.4f}  {r['total_mae']:>12.4f}  {r['oreb_mae']:>10.4f}  {', '.join(notes)}")


def print_summary(all_results):
    print(f"\n{'='*65}")
    print("  SUMMARY: Optimal values vs current")
    print(f"{'='*65}")
    print(f"  {'Parameter':<18}  {'Current':>10}  {'Best (Spread)':>14}  {'Best (Total)':>14}  {'Best (OREB)':>12}")
    print(f"  {'-'*18}  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*12}")
    for param_name, (results, default_val) in all_results.items():
        bs = min(results, key=lambda r: r["spread_mae"])["value"]
        bt = min(results, key=lambda r: r["total_mae"])["value"]
        bo = min(results, key=lambda r: r["oreb_mae"])["value"]
        print(f"  {param_name:<18}  {str(default_val):>10}  {str(bs):>14}  {str(bt):>14}  {str(bo):>12}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("NBA Parameter Justification Tool")
    print("Loading games_raw.csv and sweeping parameters...")
    print("(No API calls — uses saved data only)\n")

    # ── Part 1: Feature justification (why these variables?) ──────────────────
    print("Building baseline dataset for feature analysis...")
    baseline = build_dataset(DEFAULTS)

    justify_features(baseline, SPREAD_FEATURES, "SPREAD",     "Spread Model")
    justify_features(baseline, TOTAL_FEATURES,  "TOTAL",      "Total Model")
    justify_features(baseline, OREB_FEATURES,   "OREB_TOTAL", "OREB Model")

    # ── Part 2: Parameter sweep (why these hyperparameter values?) ─────────────
    print(f"\n\n{'#'*65}")
    print("  PART 2: HYPERPARAMETER SWEEP")
    print(f"{'#'*65}")
    print("Sweeping each parameter while holding others at current values...\n")

    all_results = {}

    for param_name in list(GRIDS.keys()):
        print(f"Sweeping {param_name} ({len(GRIDS[param_name])} values)...", flush=True)
        results = sweep_param(param_name, GRIDS[param_name], DEFAULTS)
        all_results[param_name] = (results, DEFAULTS[param_name])
        print_sweep(param_name, results, DEFAULTS[param_name])

    print_summary(all_results)
    print("\nDone. Update DEFAULTS in build_dataset.py with any improvements found above.")


if __name__ == "__main__":
    main()
