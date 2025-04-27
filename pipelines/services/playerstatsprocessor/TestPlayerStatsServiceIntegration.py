import unittest

import numpy as np
import pandas as pd

from pipelines.services.playerstatsprocessor.PlayerStatsProcessor import PlayerStatsProcessor


class TestPlayerStatsServiceIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.player_attributes = pd.DataFrame({
            'player_api_id': [1, 1, 1, 2, 2, 3],
            'date': pd.to_datetime([
                '2020-01-01', '2020-01-15', '2020-02-01',
                '2020-01-01', '2020-01-15', '2020-01-01'
            ]),
            'overall_rating': [80, 82, 81, 75, 76, 70],
            'acceleration': [70, 72, 71, 80, 81, 65],
            'strength': [85, 84, 86, 70, 72, 75],
            'aggression': [60, 62, 61, 65, 66, 70]
        })

        self.match_data = pd.DataFrame({
            'match_api_id': [101, 102, 103, 104],
            'date': pd.to_datetime([
                '2020-02-15', '2020-03-01', '2020-03-15', '2020-04-01'
            ]),
            'home_team': [10, 11, 10, 11],
            'away_team': [11, 10, 12, 10],
            'home_player_1': [1, 2, 1, 2],
            'home_player_2': [3, np.nan, 3, np.nan],
            'away_player_1': [2, 1, 3, 1],
            'away_player_2': [np.nan, 3, np.nan, 3]
        })

    def test_integration_with_pipeline(self):
        """Test that the service works as expected in the pipeline context."""
        processor = PlayerStatsProcessor()
        players_cols = ['home_player_1', 'home_player_2', 'away_player_1', 'away_player_2']

        player_stats_dict_series = self.match_data.apply(
            lambda row: processor.get_player_statistics(
                match_row=row,
                df_matches=self.match_data,
                df_player_attr=self.player_attributes,
                players=players_cols
            ),
            axis=1
        )

        player_stats_df = pd.json_normalize(player_stats_dict_series)

        self.assertEqual(len(player_stats_df), len(self.match_data))

        self.assertIn('match_api_id', player_stats_df.columns)

        for player in players_cols:
            self.assertIn(f"overall_rating_{player}", player_stats_df.columns)
            self.assertIn(f"acceleration_rating_{player}", player_stats_df.columns)
            self.assertIn(f"strength_rating_{player}", player_stats_df.columns)
            self.assertIn(f"aggression_rating_{player}", player_stats_df.columns)


if __name__ == '__main__':
    unittest.main()