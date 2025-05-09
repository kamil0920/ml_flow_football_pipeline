import unittest
import pandas as pd
import numpy as np
from pipelines.services.teamfeaturesprocessor.TeamFeaturesProcessor import TeamFeaturesProcessor

def make_test_df():
    data = {
        'match_api_id': [1000, 1001, 1002, 1003, 1004],
        'season': [2021]*5,
        'stage': [10, 11, 12, 13, 14],
        'date': pd.to_datetime(['2020-12-31', '2021-01-01', '2021-01-10', '2021-01-20', '2021-01-30']),
        'home_team': ['A', 'B', 'B', 'A', 'B', ],
        'away_team': ['B', 'A', 'A', 'B', 'A', ],
        'home_goals_shifted': [1, 2, 0, 3, 1],
        'away_goals_shifted': [0, 1, 3, 1, 2],
        'goals_conversion_rate_home': [0.5, 0.6, 0.0, 0.75, 0.33],
        'goals_conversion_rate_away': [0.0, 0.5, 0.75, 0.5, 0.66],
        'home_shots_shifted': [2, 3, 0, 4, 2],
        'away_shots_shifted': [0, 2, 4, 2, 3],
        'result_match': ['H', 'H', 'H', 'A', 'D'],
        'points_home': [3, 4, 3, 0, 3],
        'points_away': [0, 3, 0, 3, 0]
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
        is_weekend = self.result.loc[self.result['stage'] == 12, 'is_weekend'].iloc[0]
        self.assertTrue(is_weekend)

    def test_rest_period_home_stage3(self):
        row = self.result.loc[self.result['stage'] == 12].iloc[0]
        self.assertEqual(row['rest_period_home'], 9)

    def test_momentum_and_streaks(self):
        row3 = self.result.loc[self.result['stage']==14].iloc[0]
        self.assertEqual(row3['streak_home'], 3)


class TestTeamFeaturesProcessorIntegration(unittest.TestCase):
    def test_pipeline_idempotent(self):
        df = make_test_df()
        p1 = TeamFeaturesProcessor(df.copy(), window=4)
        p2 = TeamFeaturesProcessor(df.copy(), window=4)
        first = p1.process()
        second = p2.process()
        pd.testing.assert_frame_equal(first, second)

    def test_merge_correctness(self):
        df = make_test_df()
        result = TeamFeaturesProcessor(df, window=2).process()
        row2 = result.loc[result['stage']==12].iloc[0]
        self.assertAlmostEqual(row2['rolling_avg_goals_shifted_away'], 2.333, places=3)

if __name__ == '__main__':
    unittest.main()