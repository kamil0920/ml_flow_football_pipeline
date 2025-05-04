from unittest.mock import inplace

import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.linear_model import LinearRegression


def categorize_season(date: pd.Timestamp) -> str:
    month = date.month
    if month in [8, 9, 10]:
        return 'early'
    elif month in [11, 12, 1]:
        return 'mid'
    else:
        return 'late'



def calculate_slope(series: pd.Series, window: int = 5) -> List[float]:
    slopes = [np.nan] * len(series)
    for i in range(window, len(series)):
        y = series.iloc[i - window:i].values.reshape(-1, 1)
        x = np.arange(window).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        slopes[i] = float(model.coef_[0])
    return slopes


def calculate_streak(results: pd.Series) -> List[int]:
    streaks = []
    current_streak = 0
    for res in results:
        if res == 1:
            current_streak = current_streak + 1 if current_streak >= 0 else 1
        elif res == 0:
            current_streak = current_streak - 1 if current_streak <= 0 else -1
        else:
            current_streak = 0
        streaks.append(current_streak)
    return streaks


def merge_rolling(df: pd.DataFrame, stats: pd.DataFrame, location: str) -> pd.DataFrame:
    """
    Merges computed rolling stats back onto the main DataFrame for specified location.
    """
    prefix = location
    base_cols = ['team', 'season', 'stage', 'date', 'match_api_id']
    merge_cols = [c for c in stats.columns if c.startswith('roll_')]

    merged = df.merge(
        stats[base_cols + merge_cols],
        left_on=['match_api_id', f'{prefix}_team'],
        right_on=['match_api_id', 'team'],
        how='left'
    )

    rename_map = {
        f'roll_mean_{m}': f'rolling_avg_{m}_{prefix}' for m in ['goals_shifted', 'goals_conversion_rate', 'shots_shifted']
    }
    rename_map.update({
        f'roll_std_{m}': f'rolling_stability_{m}_{prefix}' for m in ['goals_shifted', 'goals_conversion_rate', 'shots_shifted']
    })
    merged = merged.rename(columns=rename_map)

    return merged.drop(columns=['team'])


class TeamFeaturesProcessor:
    """
    A class to compute and attach football match features, including rolling statistics for goals,
    conversion rates, shot on target, temporal context, form, momentum, and streaks.
    """

    def __init__(self, df: pd.DataFrame, window: int = 5,
                 gammas: Optional[List[float]] = None,
                 alphas: Optional[List[float]] = None):
        self.df = df.copy()
        self.window = window
        self.gammas = gammas or [0.25, 0.33, 0.5]
        self.alphas = alphas or [0.6, 0.8, 1.0]
        self._rating_groups = {
            'strength': 'strength_rating',
            'aggression': 'aggression_rating',
            'acceleration': 'acceleration_rating',
            'overall': 'overall_rating'
        }

    def process(self) -> pd.DataFrame:
        self.df['date'] = pd.to_datetime(self.df['date'])
        self._compute_basic_team_averages()
        self._compute_conversion_rates()
        self._compute_strength_acceleration_interactions()
        self._compute_rolling_features()
        self._compute_shot_efficiency()
        self._compute_temporal_context()
        self._compute_rest_periods()
        self._compute_momentum_and_streaks()
        return self.df

    def _compute_basic_team_averages(self) -> None:
        """
        Computes average ratings for home and away teams and their differences.
        """
        for feature, prefix in self._rating_groups.items():
            home_cols = self.df.filter(like=f"{prefix}_home").columns
            away_cols = self.df.filter(like=f"{prefix}_away").columns

            self.df[f'avg_{feature}_home'] = self.df[home_cols].mean(axis=1)
            self.df[f'avg_{feature}_away'] = self.df[away_cols].mean(axis=1)
            self.df[f'{feature}_difference'] = (
                    self.df[f'avg_{feature}_home'] - self.df[f'avg_{feature}_away']
            )

            self.df.drop(home_cols, axis=1, inplace=True)
            self.df.drop(away_cols, axis=1, inplace=True)

    def _compute_conversion_rates(self) -> None:
        """
        Calculates goal conversion rates for home and away teams.
        """
        self.df['goals_conversion_rate_home'] = np.where(
            self.df['home_shots_shifted'] > 0,
            self.df['home_goals_shifted'] / self.df['home_shots_shifted'],
            0
        )

        self.df['goals_conversion_rate_away'] = np.where(
            self.df['away_shots_shifted'] > 0,
            self.df['away_goals_shifted'] / self.df['away_shots_shifted'],
            0
        )

    def _compute_strength_acceleration_interactions(self) -> None:
        """
        Computes interaction term between average strength and acceleration.
        """
        self.df['home_strength_x_acceleration'] = (
                self.df['avg_strength_home'] * self.df['avg_acceleration_home']
        )
        self.df['away_strength_x_acceleration'] = (
                self.df['avg_strength_away'] * self.df['avg_acceleration_away']
        )

    def _compute_rolling_features(self) -> None:
        """
        Generates rolling exponential moving averages and std deviations for goals,
        conversion rates, and shots on target for each team over the specified window.
        """
        base_cols = ['season', 'stage', 'date']
        home_stats = self._prepare_team_frame('home')
        away_stats = self._prepare_team_frame('away')

        team_frame = pd.concat([home_stats, away_stats], ignore_index=True)
        team_frame = team_frame.sort_values(by=['team'] + base_cols)

        for metric in ['goals_shifted', 'goals_conversion_rate', 'shots_shifted']:
            team_frame[f'roll_mean_{metric}'] = (
                team_frame.groupby('team')[metric]
                .transform(lambda x: x.ewm(span=self.window, adjust=False).mean())
            )
            team_frame[f'roll_std_{metric}'] = (
                team_frame.groupby('team')[metric]
                .transform(lambda x: x.ewm(span=self.window, adjust=False).std())
            )

        self.df = merge_rolling(self.df, team_frame, 'home')
        self.df = merge_rolling(self.df, team_frame, 'away')

    def _prepare_team_frame(self, location: str) -> pd.DataFrame:
        """
        Extracts and renames columns for home or away team stats.
        """
        suffix = 'home' if location == 'home' else 'away'
        cols_map = {
            f'{location}_team': 'team',
            f'{location}_goals_shifted': 'goals_shifted',
            f'goals_conversion_rate_{suffix}': 'goals_conversion_rate',
            f'{location}_shots_shifted': 'shots_shifted'
        }
        df_sub = self.df[cols_map.keys()].rename(columns=cols_map)
        for key in ['season', 'stage', 'date', 'match_api_id']:
            df_sub[key] = self.df[key]
        return df_sub

    def _drop_intermediate_columns(self) -> None:
        """
        Drops columns used only for intermediate computations.
        """
        drop_patterns = [f'{grp}_{loc}' for grp in self._rating_groups for loc in ['home', 'away']]
        intermediate_cols = self.df.filter(regex='|'.join(drop_patterns)).columns.tolist()
        self.df.drop(columns=intermediate_cols, inplace=True)

    def _compute_shot_efficiency(self) -> None:
        self.df['shot_efficiency_ratio_home'] = (
                self.df['rolling_stability_goals_shifted_home'] /
                (self.df['rolling_stability_shots_shifted_home'] + 1e-5)
        )
        self.df['shot_efficiency_ratio_away'] = (
                self.df['rolling_stability_goals_shifted_away'] /
                (self.df['rolling_stability_shots_shifted_away'] + 1e-5)
        )

    def _compute_temporal_context(self) -> None:
        self.df['seasonal_context'] = self.df['date'].apply(categorize_season)
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        self.df['is_weekend'] = self.df['day_of_week'].isin(['Saturday', 'Sunday'])

    def _compute_rest_periods(self) -> None:
        self.df.sort_values(['home_team', 'date'], inplace=True)
        self.df['rest_period_home'] = (
            self.df.groupby('home_team')['date'].diff().dt.days
        )
        self.df.sort_values(['away_team', 'date'], inplace=True)
        self.df['rest_period_away'] = (
            self.df.groupby('away_team')['date'].diff().dt.days
        )


    def _compute_momentum_and_streaks(self) -> None:
        # Momentum
        self.df.sort_values(['home_team', 'date'], inplace=True)
        self.df['momentum_home'] = (
            self.df.groupby('home_team')['points_home']
            .transform(lambda s: calculate_slope(s, window=self.window))
            .shift(1)
        )
        self.df.sort_values(['away_team', 'date'], inplace=True)
        self.df['momentum_away'] = (
            self.df.groupby('away_team')['points_away']
            .transform(lambda s: calculate_slope(s, window=self.window))
            .shift(1)
        )
        self.df.sort_values(['home_team', 'date'], inplace=True)
        self.df['streak_home'] = (
            self.df.groupby('home_team')['result_match']
            .transform(calculate_streak)
            .shift(1)
        )
        self.df.sort_values(['away_team', 'date'], inplace=True)
        self.df['streak_away'] = (
            self.df.groupby('away_team')['result_match']
            .transform(calculate_streak)
            .shift(1)
        )
