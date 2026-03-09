"""
NBA Prediction Pipeline: Data Collection & Feature Engineering
==============================================================
Run this script to build the master dataset for Spread, Total, and OREB prediction.

Requirements:
    pip install nba_api pandas numpy scikit-learn xgboost lightgbm optuna statsmodels requests lxml

Usage:
    python build_dataset.py

Output:
    - data/games_raw.csv          (raw game-level data)
    - data/games_features.csv     (full feature matrix)
    - data/train_spread.csv       (training set for Spread model)
    - data/train_total.csv        (training set for Total model)
    - data/train_oreb.csv         (training set for OREB model)
    - data/prediction_features.csv (features for March 14 - April 12 games)
"""

import math
import os
import time
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# CONFIGURATION
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
DATA_DIR = "data"
ROLLING_LONG = 40    # baseline window (columns still labeled _30)
ROLLING_SHORT = 7    # recent form window (columns still labeled _10)
ELO_K = 16
ELO_HOME_BONUS = 150
ELO_INIT = 1500
ELO_CARRYOVER = 0.5
API_DELAY = 0.7      # seconds between nba_api calls

os.makedirs(DATA_DIR, exist_ok=True)


# STEP 1: PULL RAW GAME DATA
def pull_game_logs():
    """
    Pull team game logs for all seasons using nba_api.
    Returns a DataFrame with one row per team per game.
    """
    from nba_api.stats.endpoints import LeagueGameFinder

    all_games = []

    for season in SEASONS:
        print(f"Pulling game logs for {season}...")
        finder = LeagueGameFinder(
            season_nullable=season,
            season_type_nullable="Regular Season",
            league_id_nullable="00",
        )
        time.sleep(API_DELAY)
        df = finder.get_data_frames()[0]
        df["SEASON"] = season
        all_games.append(df)

    raw = pd.concat(all_games, ignore_index=True)
    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])
    raw = raw.sort_values("GAME_DATE").reset_index(drop=True)

    print(f"Total team-game rows pulled: {len(raw)}")
    return raw


def reshape_to_game_level(raw):
    """
    Convert team-level rows into game-level rows.
    Each game appears once with HOME_ and AWAY_ prefixed columns.
    """
    raw["IS_HOME"] = raw["MATCHUP"].str.contains("vs.").astype(int)

    home = raw[raw["IS_HOME"] == 1].copy()
    away = raw[raw["IS_HOME"] == 0].copy()

    stat_cols = [
        "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
        "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS",
    ]

    home_renamed = home.rename(
        columns={c: f"HOME_{c}" for c in stat_cols}
    )[["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID", "TEAM_ABBREVIATION"]
      + [f"HOME_{c}" for c in stat_cols]]
    home_renamed = home_renamed.rename(columns={
        "TEAM_ID": "HOME_TEAM_ID",
        "TEAM_ABBREVIATION": "HOME_TEAM",
    })

    away_renamed = away.rename(
        columns={c: f"AWAY_{c}" for c in stat_cols}
    )[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION"]
      + [f"AWAY_{c}" for c in stat_cols]]
    away_renamed = away_renamed.rename(columns={
        "TEAM_ID": "AWAY_TEAM_ID",
        "TEAM_ABBREVIATION": "AWAY_TEAM",
    })

    games = home_renamed.merge(away_renamed, on="GAME_ID", how="inner")

    games["SPREAD"] = games["HOME_PTS"] - games["AWAY_PTS"]
    games["TOTAL"] = games["HOME_PTS"] + games["AWAY_PTS"]
    games["OREB_TOTAL"] = games["HOME_OREB"] + games["AWAY_OREB"]

    games = games.sort_values("GAME_DATE").reset_index(drop=True)
    print(f"Game-level rows: {len(games)}")
    return games


# STEP 2: ADVANCED STATS
def compute_derived_advanced_stats(games):
    """
    Estimate advanced stats from basic box score data.
    Faster alternative to pulling BoxScoreAdvancedV3 for every game.
    """
    for prefix in ["HOME", "AWAY"]:
        fga = games[f"{prefix}_FGA"]
        fgm = games[f"{prefix}_FGM"]
        fg3m = games[f"{prefix}_FG3M"]
        ftm = games[f"{prefix}_FTM"]
        fta = games[f"{prefix}_FTA"]
        tov = games[f"{prefix}_TOV"]
        oreb = games[f"{prefix}_OREB"]

        opp = "AWAY" if prefix == "HOME" else "HOME"
        opp_dreb = games[f"{opp}_DREB"]
        opp_fga = games[f"{opp}_FGA"]
        opp_fgm = games[f"{opp}_FGM"]
        opp_fg3m = games[f"{opp}_FG3M"]
        opp_ftm = games[f"{opp}_FTM"]
        opp_fta = games[f"{opp}_FTA"]
        opp_tov = games[f"{opp}_TOV"]
        opp_oreb = games[f"{opp}_OREB"]
        dreb = games[f"{prefix}_DREB"]

        poss = fga - oreb + tov + 0.44 * fta
        opp_poss = opp_fga - opp_oreb + opp_tov + 0.44 * opp_fta

        games[f"{prefix}_POSS"] = poss
        games[f"{prefix}_PACE"] = (poss + opp_poss) / 2

        games[f"{prefix}_ORTG"] = np.where(poss > 0, (games[f"{prefix}_PTS"] / poss) * 100, np.nan)
        games[f"{prefix}_DRTG"] = np.where(opp_poss > 0, (games[f"{opp}_PTS"] / opp_poss) * 100, np.nan)
        games[f"{prefix}_NET_RTG"] = games[f"{prefix}_ORTG"] - games[f"{prefix}_DRTG"]

        games[f"{prefix}_EFG"] = np.where(fga > 0, (fgm + 0.5 * fg3m) / fga, np.nan)

        games[f"{prefix}_TOV_RATE"] = np.where(
            (fga + 0.44 * fta + tov) > 0,
            tov / (fga + 0.44 * fta + tov),
            np.nan,
        )

        games[f"{prefix}_OREB_RATE"] = np.where(
            (oreb + opp_dreb) > 0,
            oreb / (oreb + opp_dreb),
            np.nan,
        )

        games[f"{prefix}_DREB_RATE"] = np.where(
            (dreb + opp_oreb) > 0,
            dreb / (dreb + opp_oreb),
            np.nan,
        )

        games[f"{prefix}_FT_RATE"] = np.where(fga > 0, ftm / fga, np.nan)

        games[f"{prefix}_EXPECTED_MISSES"] = fga * (1 - np.where(fga > 0, fgm / fga, 0))

    return games


# STEP 2b: PLAYER AVAILABILITY SYSTEM
def pull_player_bpm_bbref(seasons):
    """
    Load player Box Plus/Minus (BPM) from locally saved Basketball Reference CSVs.
    Download from https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html
    and save as data/bbref_advanced_{year}.csv before running.
    Returns DataFrame with columns: Player, BPM, SEASON.
    """
    season_year_map = {"2021-22": 2022, "2022-23": 2023, "2023-24": 2024, "2024-25": 2025, "2025-26": 2026}
    records = []

    for season in seasons:
        year = season_year_map.get(season)
        if year is None:
            continue
        local_path = f"{DATA_DIR}/bbref_advanced_{year}.csv"
        print(f"  Loading BBRef advanced stats for {season} from {local_path}...")

        try:
            df = pd.read_csv(local_path)
            # Drop repeated header rows (present in HTML export, harmless if absent in CSV)
            df = df[df["Rk"] != "Rk"].copy()
            df["BPM"] = pd.to_numeric(df["BPM"], errors="coerce")
            # CSV uses "Team" column; HTML uses "Tm" — normalize to "Tm"
            if "Team" in df.columns and "Tm" not in df.columns:
                df = df.rename(columns={"Team": "Tm"})
            # Traded players appear once per team + a "TOT" (season totals) row.
            # Keep only the TOT row for those players.
            has_tot = set(df[df["Tm"] == "TOT"]["Player"].unique())
            df = pd.concat([
                df[(df["Player"].isin(has_tot)) & (df["Tm"] == "TOT")],
                df[~df["Player"].isin(has_tot)],
            ], ignore_index=True)
            df = df[["Player", "BPM"]].dropna(subset=["BPM"]).drop_duplicates("Player")
            df["SEASON"] = season
            records.append(df)
            print(f"    {len(df)} player BPM records for {season}")
        except Exception as exc:
            print(f"  WARNING: BBRef load failed for {season}: {exc}")

    if not records:
        return pd.DataFrame(columns=["Player", "BPM", "SEASON"])
    return pd.concat(records, ignore_index=True)


def pull_player_season_stats(seasons):
    """
    Pull per-player season-average stats from nba_api (PerGame mode).
    Provides avg MIN per player for prediction availability scoring.
    Returns DataFrame with PLAYER_ID, PLAYER_NAME, TEAM_ID, MIN, SEASON.
    """
    from nba_api.stats.endpoints import LeagueDashPlayerStats

    all_stats = []
    for season in seasons:
        print(f"  Pulling player season stats for {season}...")
        try:
            stats = LeagueDashPlayerStats(
                season=season,
                season_type_all_star="Regular Season",
                per_mode_detailed="PerGame",
            )
            time.sleep(API_DELAY)
            df = stats.get_data_frames()[0]
            df.columns = df.columns.str.upper()
            df["SEASON"] = season
            all_stats.append(df)
            print(f"    {len(df)} player records for {season}")
        except Exception as exc:
            print(f"  WARNING: LeagueDashPlayerStats failed for {season}: {exc}")

    if not all_stats:
        return pd.DataFrame()
    return pd.concat(all_stats, ignore_index=True)


def pull_player_game_logs(seasons):
    """
    Pull per-player per-game stats (including actual minutes played) from nba_api.
    Uses the bulk PlayerGameLogs endpoint — one API call per season.
    Returns DataFrame with PLAYER_ID, PLAYER_NAME, GAME_ID, TEAM_ID, MIN, SEASON.
    """
    from nba_api.stats.endpoints import PlayerGameLogs

    all_logs = []
    for season in seasons:
        print(f"  Pulling player game logs for {season}...")
        try:
            logs = PlayerGameLogs(
                season_nullable=season,
                season_type_nullable="Regular Season",
                league_id_nullable="00",
            )
            time.sleep(API_DELAY)
            df = logs.get_data_frames()[0]
            df.columns = df.columns.str.upper()
            df["SEASON"] = season
            all_logs.append(df)
            print(f"    {len(df)} player-game rows for {season}")
        except Exception as exc:
            print(f"  WARNING: PlayerGameLogs failed for {season}: {exc}")

    if not all_logs:
        return pd.DataFrame()
    return pd.concat(all_logs, ignore_index=True)


def _parse_minutes(val):
    """Parse minutes from 'MM:SS' string or numeric. Returns float."""
    try:
        if isinstance(val, str) and ":" in val:
            parts = val.split(":")
            return float(parts[0]) + float(parts[1]) / 60
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def compute_team_availability_scores(games, player_logs, player_bpm):
    """
    For each historical game, compute team availability = sum(BPM x actual_minutes)
    for all players who appeared. Adds HOME_AVAILABILITY and AWAY_AVAILABILITY to games.

    player_logs: from pull_player_game_logs  (GAME_ID, TEAM_ID, PLAYER_NAME, MIN)
    player_bpm:  from pull_player_bpm_bbref  (Player, BPM, SEASON)

    Design: BPM is a season-level metric (quality); actual minutes is the per-game
    weight (usage). sum(BPM x min) captures both who plays and how good they are.
    Players missing from BBRef get BPM=0 (replacement-level assumption).
    """
    if player_logs.empty or player_bpm.empty:
        games["HOME_AVAILABILITY"] = 0.0
        games["AWAY_AVAILABILITY"] = 0.0
        return games

    logs = player_logs.copy()
    logs["MIN_FLOAT"] = logs["MIN"].apply(_parse_minutes) if "MIN" in logs.columns else 0.0

    name_col = next((c for c in ["PLAYER_NAME", "PLAYER"] if c in logs.columns), None)
    if name_col and name_col != "PLAYER_NAME":
        logs = logs.rename(columns={name_col: "PLAYER_NAME"})
    logs["PLAYER_NAME"] = logs["PLAYER_NAME"].astype(str).str.strip()

    # PlayerGameLogs already includes GAME_DATE; merge only if it's absent.
    if "GAME_DATE" not in logs.columns:
        game_dates = games[["GAME_ID", "GAME_DATE"]].drop_duplicates()
        logs = logs.merge(game_dates, on="GAME_ID", how="left")
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    logs = logs.dropna(subset=["GAME_DATE"])
    logs = logs.sort_values(["PLAYER_NAME", "GAME_DATE"])

    logs["ROLLING_MIN"] = logs.groupby("PLAYER_NAME")["MIN_FLOAT"].transform(lambda x: x.shift(1).rolling(15, min_periods=1).mean())
    logs["ROLLING_MIN"] = logs["ROLLING_MIN"].fillna(0.0)

    bpm = player_bpm.rename(columns={"Player": "PLAYER_NAME"}).copy()
    bpm["PLAYER_NAME"] = bpm["PLAYER_NAME"].str.strip()
    bpm["SEASON"] = bpm["SEASON"].apply(lambda s: str(int(s[:4]) + 1) + chr(45) + str(int(s[:4]) + 2)[-2:])

    merged = logs.merge(bpm[["PLAYER_NAME", "SEASON", "BPM"]], on=["PLAYER_NAME", "SEASON"], how="left")
    merged["BPM"] = merged["BPM"].fillna(0.0)
    merged["CONTRIB"] = merged["BPM"] * merged["ROLLING_MIN"]

    team_avail = merged.groupby(["GAME_ID", "TEAM_ID"])["CONTRIB"].sum().reset_index()
    team_avail = team_avail.rename(columns={"CONTRIB": "AVAILABILITY_SCORE"})

    home_avail = team_avail.rename(columns={"TEAM_ID": "HOME_TEAM_ID", "AVAILABILITY_SCORE": "HOME_AVAILABILITY"})
    away_avail = team_avail.rename(columns={"TEAM_ID": "AWAY_TEAM_ID", "AVAILABILITY_SCORE": "AWAY_AVAILABILITY"})

    games = games.merge(home_avail[["GAME_ID", "HOME_TEAM_ID", "HOME_AVAILABILITY"]], on=["GAME_ID", "HOME_TEAM_ID"], how="left")
    games = games.merge(away_avail[["GAME_ID", "AWAY_TEAM_ID", "AWAY_AVAILABILITY"]], on=["GAME_ID", "AWAY_TEAM_ID"], how="left")

    games["HOME_AVAILABILITY"] = games["HOME_AVAILABILITY"].fillna(0.0)
    games["AWAY_AVAILABILITY"] = games["AWAY_AVAILABILITY"].fillna(0.0)

    return games


def pull_current_injury_report():
    """
    Fetch current NBA injury report from ESPN's public API.
    Returns a set of player display names currently listed as Out or Doubtful.
    """
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        injured = set()
        for team_entry in data.get("injuries", []):
            for item in team_entry.get("injuries", []):
                status = item.get("status", "").lower()
                if status in ("out", "doubtful"):
                    name = item.get("athlete", {}).get("displayName", "")
                    if name:
                        injured.add(name)
        print(f"  Injury report: {len(injured)} players Out/Doubtful.")
        return injured
    except Exception as exc:
        print(f"  WARNING: ESPN injury report unavailable: {exc}")
        return set()


def compute_prediction_availability(pred_df, player_season_stats, player_bpm,
                                    injured_players, team_name_to_id):
    """
    For future games, compute availability score using season-average minutes
    for each team's roster, zeroing out injured players.

    Adds HOME_AVAILABILITY and AWAY_AVAILABILITY columns to pred_df.
    This directly solves the roster-turnover problem: the score updates immediately
    when a player is traded (new team) or injured (zeroed out of minutes).
    """
    if player_season_stats.empty or player_bpm.empty:
        pred_df["HOME_AVAILABILITY"] = np.nan
        pred_df["AWAY_AVAILABILITY"] = np.nan
        return pred_df

    ps = player_season_stats.copy()
    ps.columns = ps.columns.str.upper()

    name_col = next((c for c in ["PLAYER_NAME", "PLAYER"] if c in ps.columns), None)
    if name_col and name_col != "PLAYER_NAME":
        ps = ps.rename(columns={name_col: "PLAYER_NAME"})
    ps["PLAYER_NAME"] = ps["PLAYER_NAME"].astype(str).str.strip()
    ps["MIN_FLOAT"] = ps["MIN"].apply(_parse_minutes) if "MIN" in ps.columns else 0.0

    # Use prior-season BPM to match the training approach in compute_team_availability_scores,
    # which shifts BPM forward one season to avoid end-of-season leakage.
    # For 2025-26 predictions this means using 2024-25 BPM.
    bpm = player_bpm.rename(columns={"Player": "PLAYER_NAME"}).copy()
    bpm["PLAYER_NAME"] = bpm["PLAYER_NAME"].str.strip()
    seasons_sorted = sorted(bpm["SEASON"].unique())
    prior_season = seasons_sorted[-2] if len(seasons_sorted) >= 2 else seasons_sorted[-1]
    bpm_prior = bpm[bpm["SEASON"] == prior_season]
    bpm_latest = bpm_prior.groupby("PLAYER_NAME").last().reset_index()

    ps_merged = ps.merge(bpm_latest[["PLAYER_NAME", "BPM"]], on="PLAYER_NAME", how="left")
    ps_merged["BPM"] = ps_merged["BPM"].fillna(0.0)

    # Zero out injured players' minute contribution
    ps_merged.loc[ps_merged["PLAYER_NAME"].isin(injured_players), "MIN_FLOAT"] = 0.0
    ps_merged["CONTRIB"] = ps_merged["BPM"] * ps_merged["MIN_FLOAT"]

    # Use only the latest season stats
    if "SEASON" in ps_merged.columns:
        latest_season = ps_merged["SEASON"].max()
        ps_merged = ps_merged[ps_merged["SEASON"] == latest_season]

    team_avail = ps_merged.groupby("TEAM_ID")["CONTRIB"].sum().to_dict()

    home_avail, away_avail = [], []
    for _, row in pred_df.iterrows():
        h_id = team_name_to_id.get(row["Home"])
        a_id = team_name_to_id.get(row["Away"])
        home_avail.append(team_avail.get(h_id, np.nan))
        away_avail.append(team_avail.get(a_id, np.nan))

    pred_df["HOME_AVAILABILITY"] = home_avail
    pred_df["AWAY_AVAILABILITY"] = away_avail
    return pred_df


# STEP 3: ELO RATINGS
def compute_elo_ratings(games):
    """
    Compute Elo ratings for every team, updated after each game.
    Returns the games DataFrame with home_elo and away_elo columns
    representing each team's Elo BEFORE the game is played.
    """
    elo = {}
    prev_season = None

    home_elos = []
    away_elos = []

    for _, row in games.iterrows():
        season = row["SEASON"]
        h_id = row["HOME_TEAM_ID"]
        a_id = row["AWAY_TEAM_ID"]

        if season != prev_season and prev_season is not None:
            for tid in elo:
                elo[tid] = ELO_CARRYOVER * elo[tid] + (1 - ELO_CARRYOVER) * ELO_INIT
            prev_season = season
        if prev_season is None:
            prev_season = season

        if h_id not in elo:
            elo[h_id] = ELO_INIT
        if a_id not in elo:
            elo[a_id] = ELO_INIT

        home_elos.append(elo[h_id])
        away_elos.append(elo[a_id])

        h_elo = elo[h_id] + ELO_HOME_BONUS
        a_elo = elo[a_id]
        e_home = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))

        s_home = 1.0 if row["HOME_PTS"] > row["AWAY_PTS"] else 0.0

        margin = abs(row["HOME_PTS"] - row["AWAY_PTS"])
        k_mult = min(math.log(margin + 1), 2.5)
        elo[h_id] += ELO_K * k_mult * (s_home - e_home)
        elo[a_id] += ELO_K * k_mult * ((1 - s_home) - (1 - e_home))

    games["HOME_ELO"] = home_elos
    games["AWAY_ELO"] = away_elos
    games["ELO_DIFF"] = games["HOME_ELO"] - games["AWAY_ELO"] + ELO_HOME_BONUS

    return games, elo


# STEP 4: ROLLING FEATURES
def compute_rolling_features(games):
    """
    Compute rolling averages for each team across both windows.
    CRITICAL: shift(1) ensures we only use data from PRIOR games.
    """
    roll_stats = [
        "PTS", "OREB", "DREB", "FGA", "FGM", "FG3M", "FTM", "FTA", "TOV",
        "EFG", "TOV_RATE", "OREB_RATE", "DREB_RATE", "FT_RATE",
        "ORTG", "DRTG", "NET_RTG", "PACE", "EXPECTED_MISSES",
    ]

    team_rows = []

    for _, row in games.iterrows():
        game_date = row["GAME_DATE"]
        game_id = row["GAME_ID"]

        # Home team's stats in this game
        home_record = {"TEAM_ID": row["HOME_TEAM_ID"], "GAME_DATE": game_date, "GAME_ID": game_id}
        for stat in roll_stats:
            home_record[stat] = row.get(f"HOME_{stat}", np.nan)
        home_record["PTS_ALLOWED"] = row["AWAY_PTS"]
        home_record["WIN"] = 1 if row["HOME_PTS"] > row["AWAY_PTS"] else 0
        # Defensive Four Factors: what the opponent did against this team this game
        home_record["DEF_EFG"] = row.get("AWAY_EFG", np.nan)
        home_record["DEF_TOV_RATE"] = row.get("AWAY_TOV_RATE", np.nan)
        home_record["DEF_OREB_RATE"] = row.get("AWAY_OREB_RATE", np.nan)
        home_record["DEF_FT_RATE"] = row.get("AWAY_FT_RATE", np.nan)
        home_record["IS_HOME_GAME"] = 1
        team_rows.append(home_record)

        # Away team's stats in this game
        away_record = {"TEAM_ID": row["AWAY_TEAM_ID"], "GAME_DATE": game_date, "GAME_ID": game_id}
        for stat in roll_stats:
            away_record[stat] = row.get(f"AWAY_{stat}", np.nan)
        away_record["PTS_ALLOWED"] = row["HOME_PTS"]
        away_record["WIN"] = 1 if row["AWAY_PTS"] > row["HOME_PTS"] else 0
        # Defensive Four Factors: what the opponent did against this team this game
        away_record["DEF_EFG"] = row.get("HOME_EFG", np.nan)
        away_record["DEF_TOV_RATE"] = row.get("HOME_TOV_RATE", np.nan)
        away_record["DEF_OREB_RATE"] = row.get("HOME_OREB_RATE", np.nan)
        away_record["DEF_FT_RATE"] = row.get("HOME_FT_RATE", np.nan)
        away_record["IS_HOME_GAME"] = 0
        team_rows.append(away_record)

    team_df = pd.DataFrame(team_rows).sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    # Rolling averages for offensive stats (20-game and 5-game windows)
    # min_periods=5: require at least 5 prior games before computing a rolling average.
    # This prevents noise from 1-2 game samples at season start.
    for stat in roll_stats + ["PTS_ALLOWED", "WIN"]:
        for window, suffix in [(ROLLING_LONG, "30"), (ROLLING_SHORT, "10")]:
            col_name = f"ROLL_{stat}_{suffix}"
            team_df[col_name] = (
                team_df.groupby("TEAM_ID")[stat]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=5).mean())
            )

    # Defensive Four Factors rolling averages (20-game only)
    # Lower DEF_EFG = better defense; model learns the direction from data.
    for stat in ["DEF_EFG", "DEF_TOV_RATE", "DEF_OREB_RATE", "DEF_FT_RATE"]:
        team_df[f"ROLL_{stat}_30"] = (
            team_df.groupby("TEAM_ID")[stat]
            .transform(lambda x: x.shift(1).rolling(ROLLING_LONG, min_periods=5).mean())
        )

    # Season-to-date overall win percentage (shifted)
    team_df["SEASON_WIN_PCT"] = (
        team_df.groupby("TEAM_ID")["WIN"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )

    # Home/road split win percentages (Section 3J: venue-specific records)
    # These capture teams that disproportionately over/under-perform at home vs. road
    # (e.g. Denver altitude advantage, teams with weak home crowds).
    def _home_win_pct(group):
        shifted_win_home = (group["WIN"] * group["IS_HOME_GAME"]).shift(1)
        shifted_home = group["IS_HOME_GAME"].shift(1)
        return shifted_win_home.expanding().sum() / shifted_home.expanding().sum().clip(lower=1e-9)

    def _road_win_pct(group):
        shifted_win_road = (group["WIN"] * (1 - group["IS_HOME_GAME"])).shift(1)
        shifted_road = (1 - group["IS_HOME_GAME"]).shift(1)
        return shifted_win_road.expanding().sum() / shifted_road.expanding().sum().clip(lower=1e-9)

    team_df["HOME_WIN_PCT"] = (
        team_df.groupby("TEAM_ID", group_keys=False).apply(_home_win_pct)
    )
    team_df["ROAD_WIN_PCT"] = (
        team_df.groupby("TEAM_ID", group_keys=False).apply(_road_win_pct)
    )

    # Set to NaN until the team has played at least one home/road game
    home_games_seen = team_df.groupby("TEAM_ID")["IS_HOME_GAME"].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    road_games_seen = team_df.groupby("TEAM_ID")["IS_HOME_GAME"].transform(
        lambda x: (1 - x).shift(1).expanding().sum()
    )
    team_df.loc[home_games_seen < 1, "HOME_WIN_PCT"] = np.nan
    team_df.loc[road_games_seen < 1, "ROAD_WIN_PCT"] = np.nan

    # Rest days
    team_df["PREV_GAME_DATE"] = team_df.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    team_df["REST_DAYS"] = (team_df["GAME_DATE"] - team_df["PREV_GAME_DATE"]).dt.days.clip(upper=4)
    team_df["IS_B2B"] = (team_df["REST_DAYS"] == 1).astype(int)

    return team_df


def merge_rolling_to_games(games, team_df):
    """
    Join rolling features back to the game-level DataFrame.
    Each game gets HOME_ and AWAY_ rolling stats.
    """
    roll_cols = [c for c in team_df.columns if c.startswith("ROLL_")] + [
        "SEASON_WIN_PCT", "HOME_WIN_PCT", "ROAD_WIN_PCT", "REST_DAYS", "IS_B2B",
    ]
    merge_cols = ["GAME_ID", "TEAM_ID"] + roll_cols

    home_merge = team_df[merge_cols].rename(
        columns={c: f"HOME_{c}" for c in roll_cols}
    ).rename(columns={"TEAM_ID": "HOME_TEAM_ID"})

    games = games.merge(home_merge, on=["GAME_ID", "HOME_TEAM_ID"], how="left")

    away_merge = team_df[merge_cols].rename(
        columns={c: f"AWAY_{c}" for c in roll_cols}
    ).rename(columns={"TEAM_ID": "AWAY_TEAM_ID"})

    games = games.merge(away_merge, on=["GAME_ID", "AWAY_TEAM_ID"], how="left")

    return games


# STEP 5: BUILD DIFFERENTIAL AND INTERACTION FEATURES
def build_model_features(games):
    """
    Create the final feature columns used for modeling.
    Differentials (home - away) for Spread.
    Sums (home + away) for Total and OREB.
    Interactions for OREB matchup.
    """
    # --- OFFENSIVE DIFFERENTIAL FEATURES (primarily for Spread) ---
    diff_stats_30 = ["EFG", "TOV_RATE", "OREB_RATE", "FT_RATE", "NET_RTG", "ORTG", "DRTG", "PTS"]
    for stat in diff_stats_30:
        games[f"DIFF_{stat}_30"] = (
            games[f"HOME_ROLL_{stat}_30"] - games[f"AWAY_ROLL_{stat}_30"]
        )

    diff_stats_10 = ["NET_RTG", "PTS"]
    for stat in diff_stats_10:
        games[f"DIFF_{stat}_10"] = (
            games[f"HOME_ROLL_{stat}_10"] - games[f"AWAY_ROLL_{stat}_10"]
        )

    games["DIFF_WIN_PCT"] = games["HOME_SEASON_WIN_PCT"] - games["AWAY_SEASON_WIN_PCT"]
    games["REST_DIFF"] = games["HOME_REST_DAYS"] - games["AWAY_REST_DAYS"]

    # --- DEFENSIVE FOUR FACTORS DIFFERENTIALS ---
    # A negative DIFF_DEF_EFG_30 means home team allows more efficient shots = weaker defense.
    # The model learns the sign direction from data.
    for stat in ["DEF_EFG", "DEF_TOV_RATE", "DEF_OREB_RATE", "DEF_FT_RATE"]:
        games[f"DIFF_{stat}_30"] = (
            games[f"HOME_ROLL_{stat}_30"] - games[f"AWAY_ROLL_{stat}_30"]
        )

    # Home/road split win% differential (Section 3J)
    # HOME_HOME_WIN_PCT = home team's win% in home games (renamed by merge_rolling_to_games)
    # AWAY_ROAD_WIN_PCT = away team's win% in road games
    games["DIFF_HOME_ROAD_WIN_PCT"] = (
        games["HOME_HOME_WIN_PCT"] - games["AWAY_ROAD_WIN_PCT"]
    )

    # Availability differential: positive = home team has healthier/better roster
    if "HOME_AVAILABILITY" in games.columns:
        games["AVAILABILITY_DIFF"] = games["HOME_AVAILABILITY"] - games["AWAY_AVAILABILITY"]
    else:
        games["AVAILABILITY_DIFF"] = np.nan

    # --- SUM FEATURES (primarily for Total) ---
    sum_stats_30 = ["ORTG", "DRTG", "EFG", "FT_RATE", "PACE", "PTS"]
    for stat in sum_stats_30:
        games[f"SUM_{stat}_30"] = (
            games[f"HOME_ROLL_{stat}_30"] + games[f"AWAY_ROLL_{stat}_30"]
        )

    games["SUM_PTS_10"] = games["HOME_ROLL_PTS_10"] + games["AWAY_ROLL_PTS_10"]
    games["EXPECTED_PACE"] = (games["HOME_ROLL_PACE_30"] + games["AWAY_ROLL_PACE_30"]) / 2
    games["PACE_DIFF"] = games["HOME_ROLL_PACE_30"] - games["AWAY_ROLL_PACE_30"]

    # Interaction: pace × combined eFG (for Total)
    games["PACE_X_EFG"] = games["EXPECTED_PACE"] * games["SUM_EFG_30"]

    games["SUM_B2B"] = games["HOME_IS_B2B"] + games["AWAY_IS_B2B"]

    # --- OREB-SPECIFIC FEATURES ---
    games["SUM_OREB_RATE_30"] = (
        games["HOME_ROLL_OREB_RATE_30"] + games["AWAY_ROLL_OREB_RATE_30"]
    )
    games["SUM_OREB_10"] = games["HOME_ROLL_OREB_10"] + games["AWAY_ROLL_OREB_10"]
    games["SUM_EXPECTED_MISSES_30"] = (
        games["HOME_ROLL_EXPECTED_MISSES_30"] + games["AWAY_ROLL_EXPECTED_MISSES_30"]
    )
    games["SUM_FGA_30"] = games["HOME_ROLL_FGA_30"] + games["AWAY_ROLL_FGA_30"]

    # OREB% vs opponent DREB% matchup interaction
    games["OREB_MATCHUP_HOME"] = (
        games["HOME_ROLL_OREB_RATE_30"] * (1 - games["AWAY_ROLL_DREB_RATE_30"])
    )
    games["OREB_MATCHUP_AWAY"] = (
        games["AWAY_ROLL_OREB_RATE_30"] * (1 - games["HOME_ROLL_DREB_RATE_30"])
    )
    games["OREB_MATCHUP_TOTAL"] = games["OREB_MATCHUP_HOME"] + games["OREB_MATCHUP_AWAY"]
    games["OREB_MATCHUP_DIFF"] = games["OREB_MATCHUP_HOME"] - games["OREB_MATCHUP_AWAY"]

    return games


# STEP 6: DEFINE FEATURE SETS PER TARGET
SPREAD_FEATURES = [
    "ELO_DIFF",
    # Offensive Four Factors differentials
    "DIFF_EFG_30",
    "DIFF_TOV_RATE_30",
    "DIFF_OREB_RATE_30",
    "DIFF_FT_RATE_30",
    # Defensive Four Factors differentials (what opponent does against each team)
    "DIFF_DEF_EFG_30",
    "DIFF_DEF_TOV_RATE_30",
    "DIFF_DEF_OREB_RATE_30",
    "DIFF_DEF_FT_RATE_30",
    # Efficiency and recent form
    "DIFF_NET_RTG_30",
    "DIFF_NET_RTG_10",
    "DIFF_PTS_10",
    # Record differentials
    "DIFF_WIN_PCT",
    "DIFF_HOME_ROAD_WIN_PCT",  # home record vs road record (venue-specific strength)
    # Rest and fatigue
    "REST_DIFF",
    "HOME_IS_B2B",
    "AWAY_IS_B2B",
    # Roster availability (BPM × minutes, adjusted for injuries)
    "AVAILABILITY_DIFF",
]

TOTAL_FEATURES = [
    "EXPECTED_PACE",
    "SUM_ORTG_30",
    "SUM_DRTG_30",
    "SUM_EFG_30",
    "SUM_FT_RATE_30",
    "SUM_PTS_30",
    "SUM_PTS_10",
    "PACE_X_EFG",
]

OREB_FEATURES = [
    "SUM_OREB_RATE_30",
    "OREB_MATCHUP_TOTAL",
    "SUM_EXPECTED_MISSES_30",
    "EXPECTED_PACE",
    "SUM_OREB_10",
    "SUM_FGA_30",
    "HOME_ROLL_OREB_RATE_30",
    "AWAY_ROLL_OREB_RATE_30",
    "HOME_ROLL_DREB_RATE_30",
    "AWAY_ROLL_DREB_RATE_30",
]


# STEP 7: PREPARE TRAINING AND PREDICTION SETS
def filter_late_season_games(games, n_late=10, current_season="2025-26"):
    """
    Remove each team's last n_late games of each completed season.
    Late-season games are corrupted by load management (star players resting)
    and produce misleading training signal. Only applied to completed seasons —
    the current ongoing season is excluded since its schedule is incomplete.
    A game is removed if EITHER team is within their last n_late games of that season.
    """
    completed = games[games["SEASON"] != current_season].copy()

    home_tg = completed[["GAME_ID", "GAME_DATE", "SEASON", "HOME_TEAM_ID"]].rename(
        columns={"HOME_TEAM_ID": "TEAM_ID"}
    )
    away_tg = completed[["GAME_ID", "GAME_DATE", "SEASON", "AWAY_TEAM_ID"]].rename(
        columns={"AWAY_TEAM_ID": "TEAM_ID"}
    )
    tg = pd.concat([home_tg, away_tg]).drop_duplicates(["GAME_ID", "TEAM_ID"])
    tg = tg.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"]).reset_index(drop=True)

    tg["GAME_NUM"] = tg.groupby(["TEAM_ID", "SEASON"]).cumcount() + 1
    tg["TOTAL_GAMES"] = tg.groupby(["TEAM_ID", "SEASON"])["GAME_NUM"].transform("max")
    tg["GAMES_FROM_END"] = tg["TOTAL_GAMES"] - tg["GAME_NUM"]

    late_game_ids = set(tg.loc[tg["GAMES_FROM_END"] < n_late, "GAME_ID"])

    n_before = len(games)
    games_filtered = games[~games["GAME_ID"].isin(late_game_ids)].copy()
    print(f"  Late-season filter: removed {n_before - len(games_filtered)} games "
          f"(last {n_late} per team for completed seasons)")
    return games_filtered


def prepare_train_test(games, cutoff_date="2026-03-14"):
    """
    Split into training data (before cutoff) and drop rows with NaN features.
    Filters out late-season games from completed seasons to remove load-management noise.
    """
    cutoff = pd.to_datetime(cutoff_date)
    train = games[games["GAME_DATE"] < cutoff].copy()

    train = filter_late_season_games(train)

    train_spread = train.dropna(subset=SPREAD_FEATURES + ["SPREAD"])
    train_total = train.dropna(subset=TOTAL_FEATURES + ["TOTAL"])
    train_oreb = train.dropna(subset=OREB_FEATURES + ["OREB_TOTAL"])

    print(f"Training rows - Spread: {len(train_spread)}, Total: {len(train_total)}, OREB: {len(train_oreb)}")
    return train_spread, train_total, train_oreb


# STEP 8: BUILD PREDICTION FEATURES FOR FUTURE GAMES
def build_prediction_features(predictions_csv, team_df, elo_dict, team_name_map,
                               player_season_stats=None, player_bpm=None,
                               injured_players=None):
    """
    For each game in the predictions CSV, look up each team's most recent
    rolling features and Elo rating, then compute the same differential/sum
    features used in training.

    player_season_stats: from pull_player_season_stats (for availability)
    player_bpm: from pull_player_bpm_bbref (for availability)
    injured_players: set of player names currently Out/Doubtful
    """
    pred = pd.read_csv(predictions_csv)
    pred["Date"] = pd.to_datetime(pred["Date"])
    pred = pred.sort_values("Date").reset_index(drop=True)

    # Get most recent rolling stats per team
    latest = team_df.sort_values("GAME_DATE").groupby("TEAM_ID").last()

    # Seed last-game-date tracker from training data, then update as we process
    # each prediction game in order. This gives accurate rest days throughout the
    # prediction window (not just the first day after cutoff).
    team_last_game_date = {}
    for tid, row_stats in latest.iterrows():
        if "GAME_DATE" in row_stats.index:
            team_last_game_date[tid] = pd.to_datetime(row_stats["GAME_DATE"])

    rows = []
    for _, row in pred.iterrows():
        home_name = row["Home"]
        away_name = row["Away"]

        h_id = team_name_map.get(home_name)
        a_id = team_name_map.get(away_name)

        if h_id is None or a_id is None:
            print(f"  WARNING: Could not map team name. Home='{home_name}', Away='{away_name}'")
            rows.append({})
            continue

        h_stats = latest.loc[h_id] if h_id in latest.index else None
        a_stats = latest.loc[a_id] if a_id in latest.index else None

        if h_stats is None or a_stats is None:
            rows.append({})
            continue

        feat = {
            "Date": row["Date"],
            "Home": home_name,
            "Away": away_name,
        }

        # Elo
        feat["HOME_ELO"] = elo_dict.get(h_id, ELO_INIT)
        feat["AWAY_ELO"] = elo_dict.get(a_id, ELO_INIT)
        feat["ELO_DIFF"] = feat["HOME_ELO"] - feat["AWAY_ELO"] + ELO_HOME_BONUS

        # Copy all rolling stats (includes ROLL_DEF_* and ROLL_OREB_* etc.)
        roll_cols_to_copy = [c for c in latest.columns if c.startswith("ROLL_")]
        for c in roll_cols_to_copy:
            feat[f"HOME_{c}"] = h_stats[c]
            feat[f"AWAY_{c}"] = a_stats[c]

        feat["HOME_SEASON_WIN_PCT"] = h_stats["SEASON_WIN_PCT"]
        feat["AWAY_SEASON_WIN_PCT"] = a_stats["SEASON_WIN_PCT"]

        # Home/road split win percentages
        feat["HOME_HOME_WIN_PCT"] = h_stats["HOME_WIN_PCT"] if "HOME_WIN_PCT" in h_stats.index else np.nan
        feat["HOME_ROAD_WIN_PCT"] = h_stats["ROAD_WIN_PCT"] if "ROAD_WIN_PCT" in h_stats.index else np.nan
        feat["AWAY_HOME_WIN_PCT"] = a_stats["HOME_WIN_PCT"] if "HOME_WIN_PCT" in a_stats.index else np.nan
        feat["AWAY_ROAD_WIN_PCT"] = a_stats["ROAD_WIN_PCT"] if "ROAD_WIN_PCT" in a_stats.index else np.nan

        # Rest days: use dynamically tracked last-game-date so back-to-back games
        # within the prediction window are correctly detected, not just the first day.
        h_last = team_last_game_date.get(h_id, row["Date"])
        a_last = team_last_game_date.get(a_id, row["Date"])
        feat["HOME_REST_DAYS"] = min((row["Date"] - h_last).days, 4)
        feat["AWAY_REST_DAYS"] = min((row["Date"] - a_last).days, 4)
        feat["HOME_IS_B2B"] = 1 if feat["HOME_REST_DAYS"] == 1 else 0
        feat["AWAY_IS_B2B"] = 1 if feat["AWAY_REST_DAYS"] == 1 else 0

        rows.append(feat)

        # Advance last-game-date for both teams after processing this game
        team_last_game_date[h_id] = row["Date"]
        team_last_game_date[a_id] = row["Date"]

    pred_features = pd.DataFrame(rows)

    # Compute roster availability scores for future games
    if player_season_stats is not None and player_bpm is not None:
        injured = injured_players or set()
        pred_features = compute_prediction_availability(
            pred_features, player_season_stats, player_bpm, injured, team_name_map
        )
    else:
        pred_features["HOME_AVAILABILITY"] = np.nan
        pred_features["AWAY_AVAILABILITY"] = np.nan

    # Compute the same differentials/sums as training
    pred_features = build_model_features_for_pred(pred_features)

    return pred_features


def build_model_features_for_pred(df):
    """
    Same feature engineering as build_model_features, but for prediction rows.
    """
    # Offensive differentials
    for stat in ["EFG", "TOV_RATE", "OREB_RATE", "FT_RATE", "NET_RTG", "ORTG", "DRTG", "PTS"]:
        df[f"DIFF_{stat}_30"] = df[f"HOME_ROLL_{stat}_30"] - df[f"AWAY_ROLL_{stat}_30"]
    for stat in ["NET_RTG", "PTS"]:
        df[f"DIFF_{stat}_10"] = df[f"HOME_ROLL_{stat}_10"] - df[f"AWAY_ROLL_{stat}_10"]

    df["DIFF_WIN_PCT"] = df["HOME_SEASON_WIN_PCT"] - df["AWAY_SEASON_WIN_PCT"]
    df["REST_DIFF"] = df["HOME_REST_DAYS"] - df["AWAY_REST_DAYS"]

    # Defensive Four Factors differentials
    for stat in ["DEF_EFG", "DEF_TOV_RATE", "DEF_OREB_RATE", "DEF_FT_RATE"]:
        df[f"DIFF_{stat}_30"] = df[f"HOME_ROLL_{stat}_30"] - df[f"AWAY_ROLL_{stat}_30"]

    # Home/road win% differential
    df["DIFF_HOME_ROAD_WIN_PCT"] = df["HOME_HOME_WIN_PCT"] - df["AWAY_ROAD_WIN_PCT"]

    # Availability differential
    if "HOME_AVAILABILITY" in df.columns:
        df["AVAILABILITY_DIFF"] = df["HOME_AVAILABILITY"] - df["AWAY_AVAILABILITY"]
    else:
        df["AVAILABILITY_DIFF"] = np.nan

    # Sums for Total model
    for stat in ["ORTG", "DRTG", "EFG", "FT_RATE", "PACE", "PTS"]:
        df[f"SUM_{stat}_30"] = df[f"HOME_ROLL_{stat}_30"] + df[f"AWAY_ROLL_{stat}_30"]

    df["SUM_PTS_10"] = df["HOME_ROLL_PTS_10"] + df["AWAY_ROLL_PTS_10"]
    df["EXPECTED_PACE"] = (df["HOME_ROLL_PACE_30"] + df["AWAY_ROLL_PACE_30"]) / 2
    df["PACE_DIFF"] = df["HOME_ROLL_PACE_30"] - df["AWAY_ROLL_PACE_30"]
    df["PACE_X_EFG"] = df["EXPECTED_PACE"] * df["SUM_EFG_30"]
    df["SUM_B2B"] = df["HOME_IS_B2B"] + df["AWAY_IS_B2B"]

    # OREB-specific
    df["SUM_OREB_RATE_30"] = df["HOME_ROLL_OREB_RATE_30"] + df["AWAY_ROLL_OREB_RATE_30"]
    df["SUM_OREB_10"] = df["HOME_ROLL_OREB_10"] + df["AWAY_ROLL_OREB_10"]
    df["SUM_EXPECTED_MISSES_30"] = df["HOME_ROLL_EXPECTED_MISSES_30"] + df["AWAY_ROLL_EXPECTED_MISSES_30"]
    df["SUM_FGA_30"] = df["HOME_ROLL_FGA_30"] + df["AWAY_ROLL_FGA_30"]
    df["OREB_MATCHUP_HOME"] = df["HOME_ROLL_OREB_RATE_30"] * (1 - df["AWAY_ROLL_DREB_RATE_30"])
    df["OREB_MATCHUP_AWAY"] = df["AWAY_ROLL_OREB_RATE_30"] * (1 - df["HOME_ROLL_DREB_RATE_30"])
    df["OREB_MATCHUP_TOTAL"] = df["OREB_MATCHUP_HOME"] + df["OREB_MATCHUP_AWAY"]
    df["OREB_MATCHUP_DIFF"] = df["OREB_MATCHUP_HOME"] - df["OREB_MATCHUP_AWAY"]

    return df


# TEAM NAME MAPPING
TEAM_NAME_TO_ID = {
    "Atlanta Hawks": 1610612737,
    "Boston Celtics": 1610612738,
    "Brooklyn Nets": 1610612751,
    "Charlotte Hornets": 1610612766,
    "Chicago Bulls": 1610612741,
    "Cleveland Cavaliers": 1610612739,
    "Dallas Mavericks": 1610612742,
    "Denver Nuggets": 1610612743,
    "Detroit Pistons": 1610612765,
    "Golden State Warriors": 1610612744,
    "Houston Rockets": 1610612745,
    "Indiana Pacers": 1610612754,
    "Los Angeles Clippers": 1610612746,
    "Los Angeles Lakers": 1610612747,
    "Memphis Grizzlies": 1610612763,
    "Miami Heat": 1610612748,
    "Milwaukee Bucks": 1610612749,
    "Minnesota Timberwolves": 1610612750,
    "New Orleans Pelicans": 1610612740,
    "New York Knicks": 1610612752,
    "Oklahoma City Thunder": 1610612760,
    "Orlando Magic": 1610612753,
    "Philadelphia 76ers": 1610612755,
    "Phoenix Suns": 1610612756,
    "Portland Trail Blazers": 1610612757,
    "Sacramento Kings": 1610612758,
    "San Antonio Spurs": 1610612759,
    "Toronto Raptors": 1610612761,
    "Utah Jazz": 1610612762,
    "Washington Wizards": 1610612764,
}


# MAIN PIPELINE
def main():
    print("NBA PREDICTION PIPELINE")

    # --- Step 1: Pull data ---
    print("\n[1/9] Pulling game logs from nba_api...")
    raw = pull_game_logs()
    raw.to_csv(f"{DATA_DIR}/raw_team_logs.csv", index=False)

    # --- Step 2: Reshape ---
    print("\n[2/9] Reshaping to game-level rows...")
    games = reshape_to_game_level(raw)

    # --- Step 3: Derive advanced stats from box scores ---
    print("\n[3/9] Computing derived advanced stats...")
    games = compute_derived_advanced_stats(games)
    games.to_csv(f"{DATA_DIR}/games_raw.csv", index=False)

    # --- Step 3b: Player availability system ---
    print("\n[3b/9] Pulling player BPM from Basketball Reference...")
    player_bpm = pull_player_bpm_bbref(SEASONS)

    print("\n[3b/9] Pulling player game logs from nba_api...")
    player_logs = pull_player_game_logs(SEASONS)

    print("\n[3b/9] Pulling player season stats from nba_api...")
    player_season_stats = pull_player_season_stats(SEASONS)

    print("\n[3b/9] Computing team availability scores...")
    games = compute_team_availability_scores(games, player_logs, player_bpm)

    # --- Step 4: Elo ratings ---
    print("\n[4/9] Computing Elo ratings...")
    games, final_elo = compute_elo_ratings(games)

    # --- Step 5: Rolling features ---
    print("\n[5/9] Computing rolling features (this takes a minute)...")
    team_df = compute_rolling_features(games)

    # --- Step 6: Merge rolling features to game level ---
    print("\n[6/9] Merging rolling features to games...")
    games = merge_rolling_to_games(games, team_df)

    # --- Step 7: Build differential/sum features ---
    print("\n[7/9] Building model features...")
    games = build_model_features(games)
    games.to_csv(f"{DATA_DIR}/games_features.csv", index=False)

    # --- Step 8: Split and save ---
    print("\n[8/9] Preparing train sets...")
    train_spread, train_total, train_oreb = prepare_train_test(games)

    train_spread[SPREAD_FEATURES + ["SPREAD", "GAME_DATE", "HOME_TEAM", "AWAY_TEAM"]].to_csv(
        f"{DATA_DIR}/train_spread.csv", index=False
    )
    train_total[TOTAL_FEATURES + ["TOTAL", "GAME_DATE", "HOME_TEAM", "AWAY_TEAM"]].to_csv(
        f"{DATA_DIR}/train_total.csv", index=False
    )
    train_oreb[OREB_FEATURES + ["OREB_TOTAL", "GAME_DATE", "HOME_TEAM", "AWAY_TEAM"]].to_csv(
        f"{DATA_DIR}/train_oreb.csv", index=False
    )

    # --- Step 9: Build prediction features for March 14 - April 12 ---
    print("\n[9/9] Building prediction features for future games...")
    predictions_csv = "Predictions.csv"
    if os.path.exists(predictions_csv):
        injured_players = pull_current_injury_report()
        pred_features = build_prediction_features(
            predictions_csv, team_df, final_elo, TEAM_NAME_TO_ID,
            player_season_stats=player_season_stats,
            player_bpm=player_bpm,
            injured_players=injured_players,
        )
        pred_features.to_csv(f"{DATA_DIR}/prediction_features.csv", index=False)
        print(f"Prediction features saved: {len(pred_features)} games")
    else:
        print(f"  '{predictions_csv}' not found — skipping prediction features.")

    print("\nPIPELINE COMPLETE")
    print(f"All files saved to {DATA_DIR}/")

    print(f"\nSpread features ({len(SPREAD_FEATURES)}): {SPREAD_FEATURES}")
    print(f"Total features ({len(TOTAL_FEATURES)}): {TOTAL_FEATURES}")
    print(f"OREB features ({len(OREB_FEATURES)}): {OREB_FEATURES}")


if __name__ == "__main__":
    main()
