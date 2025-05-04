import unittest
import pandas as pd
from pipelines.services.matchdataprocessor.MatchDataProcessor import MatchDataProcessor


class TestMatchDataProcessor(unittest.TestCase):
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

        self.processor = MatchDataProcessor()

    def test_create_team_view(self):
        """Test create_team_view function."""
        home_view = self.processor.create_team_view(self.match_data, 'home')

        self.assertIn('team', home_view.columns)
        self.assertIn('team_goal', home_view.columns)
        self.assertIn('opponent_goal', home_view.columns)
        self.assertIn('is_home', home_view.columns)

        self.assertEqual(home_view.loc[0, 'team'], 'TeamA')
        self.assertEqual(home_view.loc[0, 'team_goal'], 2)
        self.assertEqual(home_view.loc[0, 'opponent_goal'], 1)
        self.assertEqual(home_view.loc[0, 'is_home'], 1)

        away_view = self.processor.create_team_view(self.match_data, 'away')

        self.assertEqual(away_view.loc[0, 'team'], 'TeamB')
        self.assertEqual(away_view.loc[0, 'team_goal'], 1)
        self.assertEqual(away_view.loc[0, 'opponent_goal'], 2)
        self.assertEqual(away_view.loc[0, 'is_home'], 0)

        with self.assertRaises(ValueError):
            self.processor.create_team_view(self.match_data, 'invalid')

    def test_combine_team_views(self):
        """Test combine_team_views function."""
        home_view = self.processor.create_team_view(self.match_data, 'home')
        away_view = self.processor.create_team_view(self.match_data, 'away')

        combined = self.processor.combine_team_views(home_view, away_view)

        self.assertEqual(combined.shape[0], home_view.shape[0] + away_view.shape[0])

        team_a_rows = combined[combined['team'] == 'TeamA']
        self.assertTrue(team_a_rows['date'].is_monotonic_increasing)

        home_view_modified = home_view.copy()
        home_view_modified['extra_col'] = 1

        with self.assertRaises(ValueError):
            self.processor.combine_team_views(home_view_modified, away_view)

    def test_create_lagged_features(self):
        """Test create_lagged_features function."""
        home_view = self.processor.create_team_view(self.match_data, 'home')
        away_view = self.processor.create_team_view(self.match_data, 'away')
        combined = self.processor.combine_team_views(home_view, away_view)

        result = self.processor.create_lagged_features(
            combined,
            feature_columns=['team_goal'],
            lag_periods=1
        )

        self.assertIn('team_goal_lag1', result.columns)

        team_a_first = result[(result['team'] == 'TeamA') & (result['is_home'] == 1)].iloc[0]
        self.assertTrue(pd.isna(team_a_first['team_goal_lag1']))

        with self.assertRaises(ValueError):
            self.processor.create_lagged_features(combined, feature_columns=['invalid_col'])

    def test_process_match_data(self):
        """Test the entire process_match_data pipeline."""
        result = self.processor.process_match_data(self.match_data)

        self.assertIn('match_api_id', result.columns)
        self.assertIn('home_goals_shifted', result.columns)
        self.assertIn('away_goals_shifted', result.columns)

        # Check that original shape is preserved
        self.assertEqual(result.shape[0], self.match_data.shape[0])

        # Test with custom parameters
        custom_result = self.processor.process_match_data(
            self.match_data,
            features_to_lag=['team_goal'],
            window_size=3,
            rename_mapping={'home_team_goal_lag1': 'custom_home_goal'},
            drop_columns=['home_team_goal']
        )

        self.assertIn('custom_home_goal', custom_result.columns)


if __name__ == '__main__':
    unittest.main()