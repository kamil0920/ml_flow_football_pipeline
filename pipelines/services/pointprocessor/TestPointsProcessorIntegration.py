import unittest
import pandas as pd
import numpy as np
from pipelines.services.pointprocessor.PointsProcessor import PointsProcessor


class TestPointsProcessorIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.match_data = pd.DataFrame({
            'match_api_id': [1, 2, 3, 4, 5],
            'date': pd.to_datetime(['2020-01-01', '2020-01-08', '2020-01-15', '2020-01-22', '2020-01-29']),
            'season': ['2020', '2020', '2020', '2020', '2020'],
            'home_team': [1, 2, 1, 2, 1],
            'away_team': [2, 1, 3, 3, 2],
            'result_match': ['H', 'A', 'D', 'H', 'A']
        })

    def test_integration_with_pipeline(self):
        """Test that the service works as expected in the pipeline context."""
        processor = PointsProcessor()
        counted_points_df = self.match_data.copy()
        counted_points_df[['points_home', 'points_away', 'match_api_id']] = counted_points_df.apply(
            lambda row: processor.count_points(row, counted_points_df),
            axis=1,
            result_type='expand'
        )

        self.assertIn('points_home', counted_points_df.columns)
        self.assertIn('points_away', counted_points_df.columns)

        self.assertEqual(counted_points_df.loc[0, 'points_home'], 0)
        self.assertEqual(counted_points_df.loc[0, 'points_away'], 0)

        self.assertEqual(counted_points_df.loc[2, 'points_home'], 6)
        self.assertEqual(counted_points_df.loc[2, 'points_away'], 0)


if __name__ == '__main__':
    unittest.main()