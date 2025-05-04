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

class TestFootballFeatureEngineerIntegration(unittest.TestCase):
    def test_pipeline_idempotent(self):
        df = make_test_df()
        p1 = TeamFeaturesProcessor(df.copy(), window=3)
        p2 = TeamFeaturesProcessor(df.copy(), window=3)
        first = p1.process()
        second = p2.process()
        pd.testing.assert_frame_equal(first, second)

    def test_merge_correctness(self):
        df = make_test_df()
        result = TeamFeaturesProcessor(df, window=2).process()
        row = result.loc[result['stage'] == 2].iloc[0]
        self.assertAlmostEqual(row['rolling_avg_goals_shifted_away'], 1.0, places=3)