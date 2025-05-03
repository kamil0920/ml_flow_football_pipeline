import logging
import os

from common import (
    PYTHON,
    FlowMixin,
    configure_logging,
    packages,
)

from metaflow import (
    FlowSpec,
    card,
    current,
    environment,
    project,
    pypi_base,
    resources,
    step,
)

configure_logging()


@project(name="football")
@pypi_base(
    python=PYTHON,
    packages=packages(
        "scikit-learn",
        "pandas",
        "numpy",
        "boto3",
        "mlflow",
    ),
)
class FeaturePipeline(FlowSpec, FlowMixin):
    """Feature Pipeline.

    This pipeline preprocesses football match and player data for model training.
    """

    @card
    @environment(
        vars={
            "MLFLOW_TRACKING_URI": os.getenv(
                "MLFLOW_TRACKING_URI",
                "http://127.0.0.1:5000",
            ),
        },
    )
    @step
    def start(self):
        """Start and prepare the Feature pipeline."""
        import mlflow

        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

        logging.info("MLFLOW_TRACKING_URI: %s", self.mlflow_tracking_uri)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        self.mode = "production" if current.is_production else "development"
        logging.info("Running flow in %s mode.", self.mode)

        self.raw_match_details = self.load_match_details()
        self.raw_player_attributes = self.load_player_attributes()

        try:
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_uri}."
            raise RuntimeError(message) from e

        self.next(self.explore_data)

    @card
    @step
    def explore_data(self):
        # TODO: ad.1
        """Explore and understand the data distributions and relationships."""
        import mlflow

        logging.info("Exploring data distributions and relationships...")

        self.match_stats = {
            "row_count": len(self.raw_match_details),
            "match_count": self.raw_match_details["match_api_id"].nunique(),
            "seasons": self.raw_match_details["season"].unique().tolist(),
            "missing_values": self.raw_match_details.isna().sum().to_dict(),
            "duplicates_values": self.raw_match_details.duplicated().value_counts().tolist()
        }

        self.player_stats = {
            "row_count": len(self.raw_player_attributes),
            "player_count": self.raw_player_attributes["player_api_id"].nunique(),
            "missing_values": self.raw_player_attributes.isna().sum().to_dict(),
            "duplicates_values": self.raw_player_attributes.duplicated().value_counts().tolist()
        }

        mlflow.log_params({"raw_match_data_statistics": self.match_stats})
        mlflow.log_params({"raw_player_data_statistics": self.player_stats})

        self.next(self.basic_cleaning)

    @card
    @step
    def basic_cleaning(self):
        """Perform basic data cleaning immediately after loading."""
        import pandas as pd

        self.match_details = self.raw_match_details.copy()
        self.player_attributes = self.raw_player_attributes.copy()

        self.match_details["date"] = pd.to_datetime(self.match_details["date"])
        self.player_attributes["date"] = pd.to_datetime(self.player_attributes["date"])

        self.match_details = self.match_details.drop_duplicates()
        self.player_attributes = self.player_attributes.drop_duplicates()

        self.match_details["home_possession"] = self.match_details["home_possession"].clip(0, 100)
        self.match_details["away_possession"] = self.match_details["away_possession"].clip(0, 100)

        self.match_details["home_possession"] = self.match_details["home_possession"].clip(0, 100)
        self.match_details["away_possession"] = self.match_details["away_possession"].clip(0, 100)

        self.next(self.engineer_player_features, self.calculate_point_features, self.shift_values)

    @card
    @step
    def engineer_player_features(self):
        """Create player features."""
        import pandas as pd
        from services.playerstatsprocessor.PlayerStatsProcessor import PlayerStatsProcessor

        logging.info("Engineering player features...")
        processor = PlayerStatsProcessor()

        self.players_cols = ['{}_player_{}'.format(team, i) for team in ['home', 'away'] for i in range(1, 12)]

        player_stats_dict_series = self.match_details.apply(
            lambda row: processor.get_player_statistics(
                match_row=row,
                df_matches=self.match_details,
                df_player_attr=self.player_attributes,
                players=self.players_cols
            ),
            axis=1
        )

        self.new_player_stats_df = pd.json_normalize(player_stats_dict_series)

        self.next(self.join)

    @card
    @step
    def calculate_point_features(self):
        """Count team points."""
        import logging
        from services.pointprocessor.PointsProcessor import PointsProcessor

        logging.info("Counting team points...")
        processor = PointsProcessor()

        self.counted_points_df = self.raw_match_details.copy()
        self.counted_points_df[['points_home', 'points_away', 'match_api_id']] = self.counted_points_df.apply(
            lambda row: processor.count_points(row, self.counted_points_df),
            axis=1,
            result_type='expand'
        )

        self.next(self.join)

    @card
    @step
    def shift_values(self):
        """Shift team values."""
        import logging
        from services.matchdataprocessor.MatchDataProcessor import MatchDataProcessor

        logging.info("Shift team columns...")

        processor = MatchDataProcessor()
        self.df = self.raw_match_details.copy()
        self.lagged_df = processor.process_match_data(self.df)

        self.next(self.join)

    @step
    def join(self, inputs):
        """Join engineered features."""
        import pandas as pd
        import mlflow

        points_df = next(inp.counted_points_df for inp in inputs if hasattr(inp, 'counted_points_df'))
        player_df = next(inp.new_player_stats_df for inp in inputs if hasattr(inp, 'new_player_stats_df'))
        lagged_df = next(inp.lagged_df for inp in inputs if hasattr(inp, 'lagged_df'))
        players_cols = next(inp.players_cols for inp in inputs if hasattr(inp, 'players_cols'))

        self.feature_df = pd.merge(points_df, player_df, how='left', on='match_api_id')
        self.feature_df.drop(players_cols, axis=1, inplace=True)

        self.feature_df = pd.merge(self.feature_df, lagged_df, how='left', on='match_api_id')

        stats = {
            "row_count": len(self.feature_df),
            "match_count": self.feature_df["match_api_id"].nunique(),
            "missing_values": self.feature_df.isna().sum().to_dict(),
            "duplicates_values": self.feature_df.duplicated().value_counts().tolist()
        }

        mlflow.log_params({"match_data_statistics": str(stats)})

        print("Join complete. Final shape:", self.feature_df.shape)

        self.next(self.end)

    @step
    def end(self):
        """End the Feature pipeline."""
        import os
        import mlflow
        import logging

        output_path = "data/preprocessed/df.csv"
        parent_dir = os.path.dirname(output_path)

        os.makedirs(parent_dir, exist_ok=True)

        self.feature_df.to_csv(output_path, index=False)
        mlflow.log_artifact(output_path)

        logging.info("Saved and logged DataFrame to MLflow as artifact.")

    def load_match_details(self):
        """Load match details dataset."""
        import pandas as pd
        return pd.read_csv('data/raw/match_details.csv')

    def load_player_attributes(self):
        """Load player attributes dataset."""
        import pandas as pd
        return pd.read_csv('data/raw/player_attributes.csv')


if __name__ == "__main__":
    FeaturePipeline()
