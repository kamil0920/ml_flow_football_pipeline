import unittest
import pandas as pd
from pipelines.services.pointprocessor.PointsProcessor import PointsProcessor


class TestPointsProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        # Create sample match data
        self.match_data = pd.DataFrame({
            'match_api_id': [1, 2, 3, 4, 5],
            'date': pd.to_datetime(['2020-01-01', '2020-01-08', '2020-01-15', '2020-01-22', '2020-01-29']),
            'season': ['2020', '2020', '2020', '2020', '2020'],
            'home_team': [1, 2, 1, 2, 1],
            'away_team': [2, 1, 3, 3, 2],
            'result_match': ['H', 'A', 'D', 'H', 'A']
        })

        self.processor = PointsProcessor()

    def test_get_points(self):
        """Test get_points method."""
        # Test home team points
        self.assertEqual(self.processor.get_points('H', True), 3)  # Home win, home team
        self.assertEqual(self.processor.get_points('D', True), 1)  # Draw, home team
        self.assertEqual(self.processor.get_points('A', True), 0)  # Away win, home team

        # Test away team points
        self.assertEqual(self.processor.get_points('H', False), 0)  # Home win, away team
        self.assertEqual(self.processor.get_points('D', False), 1)  # Draw, away team
        self.assertEqual(self.processor.get_points('A', False), 3)  # Away win, away team

        # Test invalid result
        self.assertEqual(self.processor.get_points('X', True), 0)  # Invalid result

    def test_process_team_points(self):
        """Test process_team_points method."""
        self.assertEqual(
            self.processor.process_team_points(
                team=1,
                df=self.match_data,
                match_date=pd.Timestamp('2020-01-08'),
                season='2020'
            ),
            3
        )

        self.assertEqual(
            self.processor.process_team_points(
                team=1,
                df=self.match_data,
                match_date=pd.Timestamp('2020-01-15'),
                season='2020'
            ),
            6
        )

        self.assertEqual(
            self.processor.process_team_points(
                team=1,
                df=self.match_data,
                match_date=pd.Timestamp('2020-01-22'),
                season='2020'
            ),
            7
        )

        self.assertEqual(
            self.processor.process_team_points(
                team=1,
                df=self.match_data,
                match_date=pd.Timestamp('2020-01-01'),
                season='2020'
            ),
            0
        )

        self.assertEqual(
            self.processor.process_team_points(
                team=1,
                df=self.match_data,
                match_date=pd.Timestamp('2020-01-29'),
                season='2019'
            ),
            0
        )

    def test_count_points(self):
        """Test count_points method."""
        row = self.match_data.iloc[2]  # Match 3
        home_points, away_points, match_api_id = self.processor.count_points(row, self.match_data)

        self.assertEqual(home_points, 6)

        self.assertEqual(away_points, 0)

        self.assertEqual(match_api_id, 3)

        row_with_missing = pd.Series({
            'match_api_id': 999,
            'date': pd.Timestamp('2020-01-29'),
            'season': '2020',
            'home_team': 999,
            'away_team': 888
        })

        home_points, away_points, match_api_id = self.processor.count_points(row_with_missing, self.match_data)

        self.assertEqual(home_points, 0)
        self.assertEqual(away_points, 0)
        self.assertEqual(match_api_id, 999)

if __name__ == '__main__':
    unittest.main()