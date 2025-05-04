import unittest
import pandas as pd
from pipelines.services.matchdataprocessor.MatchDataProcessor import process_match_data


class TestShiftDataServiceIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.match_data = pd.DataFrame({
            'match_api_id': [1, 2, 3, 4],
            'season': ['2020', '2020', '2020', '2020'],
            'stage': [1, 2, 3, 4],
            'date': pd.to_datetime(['2020-01-01', '2020-01-08', '2020-01-15', '2020-01-22']),
            'home_team': ['TeamA', 'TeamB', 'TeamA', 'TeamB'],
            'away_team': ['TeamB', 'TeamA', 'TeamC', 'TeamA'],
            'home_team_goal': [2, 1, 3, 0],
            'away_team_goal': [1, 2, 0, 1],
            'home_shoton': [10, 8, 12, 6],
            'away_shoton': [7, 9, 5, 8],
            'home_possession': [60, 45, 55, 40],
            'away_possession': [40, 55, 45, 60],
            'result_match': ['H', 'A', 'H', 'A']
        })

    def test_integration_with_pipeline(self):
        """Test that the service works as expected in the pipeline context."""
        shifted_df = self.match_data.copy()
        shifted_df = process_match_data(shifted_df)

        expected_columns = [
            'match_api_id',
            'home_goals_shifted',
            'home_shots_shifted',
            'home_possession_shifted',
            'away_goals_shifted',
            'away_shots_shifted',
            'away_possession_shifted'
        ]

        for col in expected_columns:
            self.assertIn(col, shifted_df.columns)


if __name__ == '__main__':
    unittest.main()