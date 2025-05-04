import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple


class MatchDataProcessor:
    """Class for processing football match data to create features based on past performance."""

    def __init__(self, default_window_size: int = 5, default_lag_periods: int = 1):
        """
        Initialize the MatchDataProcessor with default parameters.

        Args:
            default_window_size: Default window size for rolling calculations
            default_lag_periods: Default number of periods to lag features
        """
        self.default_window_size = default_window_size
        self.default_lag_periods = default_lag_periods
        self.default_features_to_lag = ['team_goal', 'team_shoton', 'team_possession']
        self.default_columns_to_drop = ['team_goal', 'team_shoton', 'team_possession']
        self.default_feature_mapping = {
            'home_team_goal_lag1': 'home_goals_shifted',
            'home_team_shoton_lag1': 'home_shots_shifted',
            'home_team_possession_lag1': 'home_possession_shifted',
            'away_team_goal_lag1': 'away_goals_shifted',
            'away_team_shoton_lag1': 'away_shots_shifted',
            'away_team_possession_lag1': 'away_possession_shifted',
        }

    def create_team_view(self, df: pd.DataFrame, side: str) -> pd.DataFrame:
        """
        Create a team-centric view of match data.

        Args:
            df: DataFrame containing match data
            side: Either 'home' or 'away'

        Returns:
            DataFrame with team-centric view

        Raises:
            ValueError: If side is not 'home' or 'away'
        """
        if side not in ['home', 'away']:
            raise ValueError("side must be either 'home' or 'away'")

        prefix = f"{side}_"
        opposite = "away_" if side == "home" else "home_"

        essential_cols = ['match_api_id', 'season', 'stage', 'date']
        team_cols = [
            (f'{prefix}team', 'team'),
            (f'{prefix}team_goal', 'team_goal'),
            (f'{opposite}team_goal', 'opponent_goal'),
            (f'{prefix}shoton', 'team_shoton'),
            (f'{prefix}possession', 'team_possession')
        ]

        select_cols = essential_cols + [col[0] for col in team_cols]
        team_df = df[select_cols].copy()
        col_mapping = {old: new for old, new in team_cols}
        team_df = team_df.rename(columns=col_mapping)
        team_df['is_home'] = 1 if side == 'home' else 0

        return team_df

    def combine_team_views(self, home_df: pd.DataFrame, away_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine home and away team views into a unified team-centric dataset.

        Args:
            home_df: DataFrame with home team view
            away_df: DataFrame with away team view

        Returns:
            Combined DataFrame

        Raises:
            ValueError: If home and away DataFrames have different columns
        """
        if set(home_df.columns) != set(away_df.columns):
            raise ValueError("Home and away DataFrames must have the same columns")

        team_df = pd.concat([home_df, away_df], ignore_index=True)

        team_df = team_df.sort_values(
            by=['team', 'season', 'stage', 'date']
        ).reset_index(drop=True)

        return team_df

    # def fill_nan_calculate_rolling_metrics(
    #         self,
    #         series: pd.Series,
    #         window_size: int = None,
    #         min_periods: int = 1
    # ) -> pd.Series:
    #     """
    #     Calculate rolling metrics, handling NaN and zero values.
    #
    #     Args:
    #         series: Series to process
    #         window_size: Size of rolling window
    #         min_periods: Minimum number of observations required
    #
    #     Returns:
    #         Processed series with rolling metrics
    #     """
    #     if window_size is None:
    #         window_size = self.default_window_size
    #
    #     mask = series.isna() | (series == 0)
    #     temp_series = series.mask(mask, np.nan)
    #
    #     rolled_values = temp_series.rolling(
    #         window=window_size,
    #         min_periods=min_periods
    #     ).mean()
    #
    #     result = series.copy()
    #     result = result.mask(mask, rolled_values)
    #
    #     return result

    def _process_feature(
            self,
            df: pd.DataFrame,
            feature: str,
            groupby_column: str,
            lag_periods: int,
            window_size: int
    ) -> pd.DataFrame:
        """
        Process a single feature to create lagged values.

        Args:
            df: DataFrame to process
            feature: Feature column name
            groupby_column: Column to group by
            lag_periods: Number of periods to lag
            window_size: Size of rolling window

        Returns:
            DataFrame with processed feature
        """
        lagged_col = f"{feature}_lag{lag_periods}"

        df[lagged_col] = df.groupby(groupby_column)[feature].shift(lag_periods)

        # df[lagged_col] = df.groupby(groupby_column)[lagged_col].transform(
        #     lambda x: self.fill_nan_calculate_rolling_metrics(x, window_size=window_size)
        # )

        return df

    def create_lagged_features(
            self,
            df: pd.DataFrame,
            feature_columns: List[str] = None,
            groupby_column: str = 'team',
            lag_periods: int = None,
            window_size: int = None
    ) -> pd.DataFrame:
        """
        Create lagged features for specified columns.

        Args:
            df: DataFrame to process
            feature_columns: List of feature column names
            groupby_column: Column to group by
            lag_periods: Number of periods to lag
            window_size: Size of rolling window

        Returns:
            DataFrame with lagged features

        Raises:
            ValueError: If any feature column is not in the DataFrame
        """
        if feature_columns is None:
            feature_columns = self.default_features_to_lag

        if lag_periods is None:
            lag_periods = self.default_lag_periods

        if window_size is None:
            window_size = self.default_window_size

        if not all(col in df.columns for col in feature_columns):
            missing = [col for col in feature_columns if col not in df.columns]
            raise ValueError(f"Columns not found in DataFrame: {missing}")

        result_df = df.copy()

        for feature in feature_columns:
            result_df = self._process_feature(
                result_df,
                feature,
                groupby_column,
                lag_periods,
                window_size
            )

        return result_df

    def merge_team_features(
            self,
            base_df: pd.DataFrame,
            team_df: pd.DataFrame,
            side: str,
            join_columns: List[str] = ['match_api_id', 'team']
    ) -> pd.DataFrame:
        """
        Merge team features back into base DataFrame.

        Args:
            base_df: Base DataFrame
            team_df: Team features DataFrame
            side: Either 'home' or 'away'
            join_columns: Columns to join on

        Returns:
            Merged DataFrame

        Raises:
            ValueError: If side is not 'home' or 'away'
        """
        if side not in ['home', 'away']:
            raise ValueError("side must be either 'home' or 'away'")

        is_side = team_df['is_home'] == (1 if side == 'home' else 0)
        side_data = team_df[is_side][join_columns].copy()

        lag_columns = [col for col in team_df.columns if '_lag' in col]

        if not lag_columns:
            return base_df

        columns_needed = join_columns + lag_columns

        side_data = side_data.merge(
            team_df[columns_needed],
            on=join_columns,
            how='left'
        )

        rename_cols = {col: f"{side}_{col}" for col in lag_columns}
        side_data = side_data.rename(columns=rename_cols)

        result_df = base_df.merge(
            side_data,
            on='match_api_id',
            how='left',
            suffixes=('', f'_{side}')
        )

        return result_df.drop(['team'], axis=1)

    def cleanup_feature_names(
            self,
            df: pd.DataFrame,
            feature_mapping: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Clean up feature names for better readability.

        Args:
            df: DataFrame to process
            feature_mapping: Mapping of old to new feature names

        Returns:
            DataFrame with cleaned feature names
        """
        result_df = df.copy()

        if feature_mapping is None:
            feature_mapping = self.default_feature_mapping

        cols_to_rename = {k: v for k, v in feature_mapping.items() if k in result_df.columns}

        if cols_to_rename:
            result_df = result_df.rename(columns=cols_to_rename)

        return result_df

    def drop_redundant_columns(
            self,
            df: pd.DataFrame,
            columns_to_drop: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Drop redundant columns from the DataFrame.

        Args:
            df: DataFrame to clean up
            columns_to_drop: List of columns to drop

        Returns:
            DataFrame with redundant columns removed
        """
        result_df = df.copy()

        if columns_to_drop is None:
            columns_to_drop = [
                'home_shoton', 'home_team_goal', 'home_possession',
                'away_shoton', 'away_team_goal', 'away_possession',
            ]

        cols_to_drop = [col for col in columns_to_drop if col in result_df.columns]

        if cols_to_drop:
            result_df = result_df.drop(columns=cols_to_drop)

        return result_df

    def process_match_data(
            self,
            data: pd.DataFrame,
            features_to_lag: Optional[List[str]] = None,
            window_size: int = None,
            rename_mapping: Optional[Dict[str, str]] = None,
            drop_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Process match data to create features based on past performance.

        This function handles the entire pipeline of processing match data into
        a format suitable for predictive modeling, by creating team-centric views
        and calculating lagged performance metrics.

        Args:
            data: DataFrame with match data
            features_to_lag: List of features to create lagged versions of
            window_size: Size of rolling window
            rename_mapping: Mapping of old to new feature names
            drop_columns: List of columns to drop

        Returns:
            DataFrame with processed match data

        Raises:
            Exception: If any error occurs during processing
        """
        df = data.copy()
        if features_to_lag is None:
            features_to_lag = self.default_features_to_lag

        if window_size is None:
            window_size = self.default_window_size

        if drop_columns is None:
            drop_columns = self.default_columns_to_drop

        original_cols = set(df.columns)

        try:
            home_df = self.create_team_view(df, 'home')
            away_df = self.create_team_view(df, 'away')

            team_df = self.combine_team_views(home_df, away_df)

            team_df_with_lags = self.create_lagged_features(
                team_df,
                features_to_lag,
                window_size=window_size
            )

            result_df = df.copy()
            result_df = self.merge_team_features(result_df, team_df_with_lags, 'home')
            result_df = self.merge_team_features(result_df, team_df_with_lags, 'away')
            result_df = self.cleanup_feature_names(result_df, rename_mapping)
            result_df = self.drop_redundant_columns(result_df, drop_columns)

            new_cols = [col for col in result_df.columns if col not in original_cols]
            cols_to_return = ['match_api_id'] + new_cols
            return result_df[cols_to_return]

        except Exception as e:
            print(f"Error processing match data: {str(e)}")
            raise

def process_match_data(
        df: pd.DataFrame,
        features_to_lag: Optional[List[str]] = None,
        window_size: int = 5,
        rename_mapping: Optional[Dict[str, str]] = None,
        drop_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Backward compatibility function that uses the MatchDataProcessor class.
    """
    processor = MatchDataProcessor(default_window_size=window_size)
    return processor.process_match_data(
        df,
        features_to_lag=features_to_lag,
        rename_mapping=rename_mapping,
        drop_columns=drop_columns
    )