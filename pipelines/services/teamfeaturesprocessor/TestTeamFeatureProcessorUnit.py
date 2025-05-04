import unittest
import pandas as pd
import numpy as np
from pipelines.services.teamfeaturesprocessor.TeamFeaturesProcessor import TeamFeaturesProcessor


def make_test_df():
    data = {
        'match_api_id': [1000, 1001, 1002],
        'season': [2021, 2021, 2021],
        'stage': [1, 2, 3],
        'date': pd.to_datetime(['2021-01-01', '2021-01-10', '2021-01-20']),
        'home_team': ['A', 'B', 'A'],
        'away_team': ['B', 'A', 'B'],
        'home_goals_shifted': [1, 2, 0],
        'away_goals_shifted': [0, 1, 3],
        'goals_conversion_rate_home': [0.5, 0.6, 0.0],
        'goals_conversion_rate_away': [0.0, 0.5, 0.75],
        'home_shots_shifted': [2, 3, 0],
        'away_shots_shifted': [0, 2, 4],
        'result_match': [1, 0, 1],
        'points_home': [3, 0, 3],
        'points_away': [0, 3, 0]
    }
    return pd.DataFrame(data)


class TestTeamFeaturesProcessorUnit(unittest.TestCase):
    def setUp(self):
        self.df = make_test_df()
        self.engineer = TeamFeaturesProcessor(self.df, window=2)
        self.result = self.engineer.process()

    def test_columns_created(self):
        expected_cols = [
            'rolling_avg_goals_shifted_home', 'rolling_stability_goals_shifted_home',
            'rolling_avg_goals_conversion_rate_home', 'rolling_stability_goals_conversion_rate_home',
            'rolling_avg_shots_shifted_home', 'rolling_stability_shots_shifted_home',
            'rolling_avg_goals_shifted_away', 'rolling_stability_goals_shifted_away',
            'rolling_avg_goals_conversion_rate_away', 'rolling_stability_goals_conversion_rate_away',
            'rolling_avg_shots_shifted_away', 'rolling_stability_shots_shifted_away',
            'shot_efficiency_ratio_home', 'shot_efficiency_ratio_away',
            'seasonal_context', 'day_of_week', 'is_weekend',
            'rest_period_home', 'rest_period_away',
            'momentum_home', 'momentum_away',
            'streak_home', 'streak_away'
        ]
        for col in expected_cols:
            self.assertIn(col, self.result.columns)

    def test_seasonal_and_weekend_flags(self):
        self.assertTrue((self.result['seasonal_context'] == 'mid').all())
        is_weekend = self.result.loc[self.result['stage'] == 2, 'is_weekend'].iloc[0]
        self.assertTrue(is_weekend)

    def test_rest_period_home_stage3(self):
        row = self.result.loc[self.result['stage'] == 3].iloc[0]
        self.assertEqual(row['rest_period_home'], 19)
