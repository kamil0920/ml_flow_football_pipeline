import pandas as pd


def get_points(result: str, is_home_team: bool) -> int:
    """Return points earned by a team based on match result and team side."""
    points_mapping = {
        'H': (3, 0),  # (home, away)
        'D': (1, 1),
        'A': (0, 3)
    }
    home_points, away_points = points_mapping.get(result, (0, 0))
    return home_points if is_home_team else away_points


def process_team_points(team: str, df: pd.DataFrame, match_date: pd.Timestamp, season: str) -> int:
    """Compute cumulative points for a team up to a specific match date in a given season."""
    past_matches = df.query(
        "(home_team == @team or away_team == @team) and (season == @season) and (date < @match_date)"
    )

    if past_matches.empty:
        return 0

    return past_matches.apply(
        lambda row: get_points(row['result_match'], row['home_team'] == team),
        axis=1
    ).sum()


def count_points(row: pd.Series, df: pd.DataFrame) -> tuple[int, int, int]:
    """Calculate points for both home and away teams prior to the current match."""
    match_date = row['date']
    season = row['season']
    home_team = row['home_team']
    away_team = row['away_team']

    home_points = process_team_points(home_team, df, match_date, season)
    away_points = process_team_points(away_team, df, match_date, season)

    return home_points, away_points, row['match_api_id']