import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union


class PlayerStatsProcessor:
    """Class for processing player statistics from football match data."""

    def __init__(self, default_n_previous: int = 10, default_alpha: float = 0.7):
        """
        Initialize the PlayerStatsProcessor with default parameters.

        Args:
            default_n_previous: Default number of previous matches to consider for player ratings
            default_alpha: Default decay factor for exponential moving average (higher = more weight to recent matches)
        """
        self.default_n_previous = default_n_previous
        self.default_alpha = default_alpha
        self.player_attributes = [
            'overall_rating',
            'acceleration',
            'strength',
            'aggression'
        ]

    def get_player_ratings(
            self,
            player_id: Union[int, float],
            match_date: pd.Timestamp,
            df_player_attr: pd.DataFrame,
            n_previous: Optional[int] = None,
            alpha: Optional[float] = None
    ) -> Tuple[float, float, float, float]:
        """
        Get player ratings from previous matches using exponential moving average.

        Args:
            player_id: ID of the player
            match_date: Date of the match to get ratings for
            df_player_attr: DataFrame with player attributes
            n_previous: Number of previous matches to consider
            alpha: Decay factor for EMA (higher = more weight to recent matches)

        Returns:
            Tuple containing (overall_rating, acceleration_rating, strength_rating, aggression_rating)
        """
        if n_previous is None:
            n_previous = self.default_n_previous

        if alpha is None:
            alpha = self.default_alpha

        if pd.isna(player_id):
            return np.nan, np.nan, np.nan, np.nan

        try:
            filtered = df_player_attr[(df_player_attr['player_api_id'] == player_id)]

            if filtered.empty:
                return np.nan, np.nan, np.nan, np.nan

            sorted_subset = filtered.sort_values(by='date')

            ratings = []
            for attribute in self.player_attributes:
                if attribute in sorted_subset.columns:
                    rating = sorted_subset[attribute].ewm(alpha=alpha, adjust=False).mean().iloc[-1]
                    ratings.append(rating)
                else:
                    ratings.append(np.nan)

            while len(ratings) < 4:
                ratings.append(np.nan)

            return tuple(ratings[:4])

        except Exception as e:
            print(f"Error calculating player ratings: {str(e)}")
            return np.nan, np.nan, np.nan, np.nan

    def get_player_id(
            self,
            row: pd.Series,
            player: str,
            team_type: str,
            df_matches: pd.DataFrame
    ) -> Union[int, float]:
        """
        Get player ID with fallback to most common player in previous matches.

        Args:
            row: DataFrame row containing match information
            player: Player column name (e.g., 'home_player_1')
            team_type: Either 'home' or 'away'
            df_matches: DataFrame with match data

        Returns:
            Player ID or np.nan if not found

        Raises:
            ValueError: If team_type is not 'home' or 'away'
        """
        if team_type not in ['home', 'away']:
            raise ValueError("team_type must be either 'home' or 'away'")

        # First try to get player ID directly from the row
        player_id = row[player]

        if pd.notna(player_id):
            return player_id

        try:
            # Get the team ID and player position number
            team_id = row[f"{team_type}_team"]

            # Extract player position number (e.g., "1" from "home_player_1")
            player_position = player.split('_')[-1]

            # Find all matches for this team (both home and away)
            team_matches = df_matches[(df_matches['home_team'] == team_id) |
                                      (df_matches['away_team'] == team_id)]

            # Filter to only include matches before the current date
            current_match_date = row["date"]
            past_matches = team_matches[team_matches["date"] < current_match_date]

            if past_matches.empty:
                return np.nan

            # Create a list to store player IDs
            player_ids = []

            # For each match, determine if the team was home or away and get the appropriate player ID
            for _, match in past_matches.iterrows():
                # Determine if the team was home or away in this match
                match_team_type = 'home' if match['home_team'] == team_id else 'away'

                # Construct the appropriate player column name
                match_player_col = f"{match_team_type}_player_{player_position}"

                # Get the player ID if it exists
                if match_player_col in match and pd.notna(match[match_player_col]):
                    player_ids.append(match[match_player_col])

            if not player_ids:
                return np.nan

            from collections import Counter
            most_common_id = Counter(player_ids).most_common(1)[0][0]

            return most_common_id

        except Exception as e:
            print(f"Error finding player ID: {str(e)}")
            return np.nan


    def get_player_statistics(
            self,
            match_row: pd.Series,
            df_matches: pd.DataFrame,
            df_player_attr: pd.DataFrame,
            players: List[str],
            n_previous: Optional[int] = None,
            alpha: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Extract player statistics for all players in a match.

        Args:
            match_row: DataFrame row containing match information
            df_matches: DataFrame with match data
            df_player_attr: DataFrame with player attributes
            players: List of player column names (e.g., ['home_player_1', 'home_player_2', ...])
            n_previous: Number of previous matches to consider
            alpha: Decay factor for EMA

        Returns:
            Dictionary with player statistics for all players
        """
        if n_previous is None:
            n_previous = self.default_n_previous

        if alpha is None:
            alpha = self.default_alpha

        player_stats_dict = {}
        match_date = match_row['date']
        player_stats_dict['match_api_id'] = match_row['match_api_id']

        for player in players:
            team_type = 'home' if 'home' in player else 'away'

            try:
                player_id = self.get_player_id(
                    row=match_row,
                    player=player,
                    team_type=team_type,
                    df_matches=df_matches
                )

                overall_rating, acceleration_rating, strength_rating, aggression_rating = self.get_player_ratings(
                    player_id=player_id,
                    match_date=match_date,
                    df_player_attr=df_player_attr,
                    n_previous=n_previous,
                    alpha=alpha
                )

                attribute_names = [
                    f"overall_rating_{player}",
                    f"acceleration_rating_{player}",
                    f"strength_rating_{player}",
                    f"aggression_rating_{player}"
                ]

                ratings = [overall_rating, acceleration_rating, strength_rating, aggression_rating]
                for name, rating in zip(attribute_names, ratings):
                    player_stats_dict[name] = rating if pd.notna(rating) else np.nan

            except Exception as e:
                print(f"Error processing player {player}: {str(e)}")
                for prefix in ['overall_rating', 'acceleration_rating', 'strength_rating', 'aggression_rating']:
                    player_stats_dict[f"{prefix}_{player}"] = np.nan

        return player_stats_dict