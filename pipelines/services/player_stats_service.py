import pandas as pd
import numpy as np


def _get_player_overall_rating_from_previous_N_last_(player_id, match_date, df_player_attr, n_previous=10, alpha=0.7):
    """
    Get player ratings from previous matches using exponential moving average.

    Parameters:
    - player_id: ID of the player
    - match_date: Date of the match to get ratings for
    - df_player_attr: DataFrame with player attributes
    - n_previous: Number of previous matches to consider
    - alpha: Decay factor for EMA (higher = more weight to recent matches)
    """
    filtered = df_player_attr[
        (df_player_attr['player_api_id'] == player_id) &
        (df_player_attr['date'] < match_date)
        ]

    if filtered.empty:
        return np.nan, np.nan, np.nan, np.nan

    filtered_subset = filtered.sort_values(by='date').tail(n_previous)

    latest_rating = filtered_subset['overall_rating'].ewm(alpha=alpha, adjust=False).mean().iloc[-1]
    acceleration_rating = filtered_subset['acceleration'].ewm(alpha=alpha, adjust=False).mean().iloc[-1]
    strength_rating = filtered_subset['strength'].ewm(alpha=alpha, adjust=False).mean().iloc[-1]
    aggression_rating = filtered_subset['aggression'].ewm(alpha=alpha, adjust=False).mean().iloc[-1]

    return latest_rating, acceleration_rating, strength_rating, aggression_rating


def _get_player_id_for_team_(row, player, team_type, df_matches):
    """Get player ID with fallback to most common player in previous matches."""
    player_id = row[player]

    if pd.notna(player_id):
        return player_id

    team_id = row[f"{team_type}_team"]
    team_matches = df_matches[df_matches[f"{team_type}_team"] == team_id]

    current_match_date = row["date"]
    col_values = team_matches[
        team_matches["date"] < current_match_date
    ][player].dropna()

    if col_values.empty:
        return np.nan

    fallback_id = col_values.value_counts().idxmax()
    return fallback_id

def get_player_stat(match_row, df_matches, df_player_attr, players, n_previous=10, alpha=0.7):
    """Extract player statistics for all players in a match."""
    player_stats_dict = {}
    match_date = match_row['date']
    player_stats_dict['match_api_id'] = match_row['match_api_id']

    for player in players:
        team_type = 'home' if 'home' in player else 'away'

        player_id = _get_player_id_for_team_(
            row=match_row,
            player=player,
            team_type=team_type,
            df_matches=df_matches
        )

        overall_rating, acceleration_rating, strength_rating, aggression_rating = _get_player_overall_rating_from_previous_N_last_(
            player_id=player_id,
            match_date=match_date,
            df_player_attr=df_player_attr,
            n_previous=n_previous,
            alpha=alpha

        )

        rating_col_name = f"overall_rating_{player}"
        acceleration_rating_col_name = f"acceleration_rating_{player}"
        strength_rating_col_name = f"strength_rating_{player}"
        aggression_rating_col_name = f"aggression_rating_{player}"

        player_stats_dict[rating_col_name] = overall_rating if pd.notna(overall_rating) else np.nan
        player_stats_dict[acceleration_rating_col_name] = acceleration_rating if pd.notna(acceleration_rating) else np.nan
        player_stats_dict[strength_rating_col_name] = strength_rating if pd.notna(strength_rating) else np.nan
        player_stats_dict[aggression_rating_col_name] = aggression_rating if pd.notna(aggression_rating) else np.nan

    return player_stats_dict