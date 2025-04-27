import unittest
import pandas as pd
import numpy as np
from pipelines.services.playerstatsprocessor.PlayerStatsProcessor import PlayerStatsProcessor


class TestPlayerStatsProcessor(unittest.TestCase):
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
            'away_player_2': [5, 3, np.nan, 3]
        })

        self.processor = PlayerStatsProcessor()

    def test_get_player_ratings(self):
        """Test get_player_ratings method."""
        match_date = pd.Timestamp('2020-02-15')
        ratings = self.processor.get_player_ratings(
            player_id=1,
            match_date=match_date,
            df_player_attr=self.player_attributes
        )

        self.assertEqual(len(ratings), 4)

        for rating in ratings:
            self.assertFalse(pd.isna(rating))

        ratings = self.processor.get_player_ratings(
            player_id=999,
            match_date=match_date,
            df_player_attr=self.player_attributes
        )

        for rating in ratings:
            self.assertTrue(pd.isna(rating))

        ratings = self.processor.get_player_ratings(
            player_id=np.nan,
            match_date=match_date,
            df_player_attr=self.player_attributes
        )

        for rating in ratings:
            self.assertTrue(pd.isna(rating))

    def test_get_player_id(self):
        """Test get_player_id method."""
        row = self.match_data.iloc[0]
        player_id = self.processor.get_player_id(
            row=row,
            player='home_player_1',
            team_type='home',
            df_matches=self.match_data
        )

        self.assertEqual(player_id, 1)

        row = self.match_data.iloc[1]
        player_id = self.processor.get_player_id(
            row=row,
            player='home_player_2',
            team_type='home',
            df_matches=self.match_data
        )

        self.assertEqual(player_id, 5)

        with self.assertRaises(ValueError):
            self.processor.get_player_id(
                row=row,
                player='home_player_1',
                team_type='invalid',
                df_matches=self.match_data
            )

    def test_get_player_statistics(self):
        """Test get_player_statistics method."""
        row = self.match_data.iloc[0]
        players = ['home_player_1', 'home_player_2', 'away_player_1', 'away_player_2']

        stats = self.processor.get_player_statistics(
            match_row=row,
            df_matches=self.match_data,
            df_player_attr=self.player_attributes,
            players=players
        )

        self.assertEqual(stats['match_api_id'], 101)

        for player in players:
            self.assertIn(f"overall_rating_{player}", stats)
            self.assertIn(f"acceleration_rating_{player}", stats)
            self.assertIn(f"strength_rating_{player}", stats)
            self.assertIn(f"aggression_rating_{player}", stats)

        self.assertFalse(pd.isna(stats['overall_rating_home_player_1']))

        self.assertTrue(pd.isna(stats['overall_rating_away_player_2']))

if __name__ == '__main__':
    unittest.main()