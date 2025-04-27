import pandas as pd
from typing import Tuple, Dict, Union, Optional


class PointsProcessor:
    """Class for processing team points from football match data."""

    def __init__(self):
        """Initialize the PointsProcessor with default parameters."""
        self.points_mapping = {
            'H': (3, 0),  # (home, away)
            'D': (1, 1),
            'A': (0, 3)
        }

    def get_points(self, result: str, is_home_team: bool) -> int:
        """
        Return points earned by a team based on match result and team side.

        Args:
            result: Match result ('H' for home win, 'D' for draw, 'A' for away win)
            is_home_team: Whether the team is the home team

        Returns:
            Points earned by the team (0, 1, or 3)
        """
        try:
            home_points, away_points = self.points_mapping.get(result, (0, 0))
            return home_points if is_home_team else away_points
        except Exception as e:
            print(f"Error calculating points: {str(e)}")
            return 0

    def process_team_points(
            self,
            team: Union[str, int],
            df: pd.DataFrame,
            match_date: pd.Timestamp,
            season: str
    ) -> int:
        """
        Compute cumulative points for a team up to a specific match date in a given season.

        Args:
            team: Team identifier
            df: DataFrame with match data
            match_date: Date of the match to calculate points up to
            season: Season identifier

        Returns:
            Total points earned by the team in the season up to the match date
        """
        try:
            past_matches = df.query(
                "(home_team == @team or away_team == @team) and (season == @season) and (date < @match_date)"
            )

            if past_matches.empty:
                return 0

            return past_matches.apply(
                lambda row: self.get_points(row['result_match'], row['home_team'] == team),
                axis=1
            ).sum()
        except Exception as e:
            print(f"Error processing team points: {str(e)}")
            return 0

    def count_points(self, row: pd.Series, df: pd.DataFrame) -> Tuple[int, int, int]:
        """
        Calculate points for both home and away teams prior to the current match.

        Args:
            row: DataFrame row containing match information
            df: DataFrame with match data

        Returns:
            Tuple containing (home_points, away_points, match_api_id)
        """
        try:
            match_date = row['date']
            season = row['season']
            home_team = row['home_team']
            away_team = row['away_team']

            home_points = self.process_team_points(home_team, df, match_date, season)
            away_points = self.process_team_points(away_team, df, match_date, season)

            return home_points, away_points, row['match_api_id']
        except Exception as e:
            print(f"Error counting points: {str(e)}")
            return 0, 0, row.get('match_api_id', 0)