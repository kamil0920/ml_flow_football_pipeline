import logging
import os
from pathlib import Path

from common import (
    PYTHON,
    FlowMixin,
    configure_logging,
    packages,
)

from metaflow import (
    FlowSpec,
    Parameter,
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
        "matplotlib",
        "seaborn",
        "boto3",
        "mlflow",
    ),
)
class FeaturePipeline(FlowSpec, FlowMixin):
    """Feature Pipeline.

    This pipeline preprocesses football match and player data for model training.
    """

    @card
    @step
    def start(self):
        """Start and prepare the Feature pipeline."""

        self.raw_match_details = self.load_match_details()
        self.raw_player_attributes = self.load_player_attributes()

        self.next(self.explore_data)

    @card
    @step
    def explore_data(self):
        """Explore and understand the data distributions and relationships."""

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

        mlflow.log_params({"data_quality": self.data_quality_metrics})

        self.next(self.basic_cleaning)

    @card
    @step
    def basic_cleaning(self):
        """Perform basic data cleaning immediately after loading."""
        import pandas as pd

        self.match_details = self.raw_match_details.copy()
        self.player_attributes = self.raw_player_attributes.copy()

        self.match_details["date"] = pd.to_datetime(self.self.match_details["date"])
        self.player_attributes["date"] = pd.to_datetime(self.self.player_attributes["date"])

        self.self.match_details = self.self.match_details.drop_duplicates()
        self.self.player_attributes = self.self.player_attributes.drop_duplicates()

        self.match_details["home_possession"] = self.match_details["home_possession"].clip(0, 100)
        self.match_details["away_possession"] = self.match_details["away_possession"].clip(0, 100)

        self.match_details["home_possession"] = self.match_details["home_possession"].clip(0, 100)
        self.match_details["away_possession"] = self.match_details["away_possession"].clip(0, 100)

        self.next(self.engineer_player_features)

    @card
    @step
    def engineer_player_features(self):
        """Create player features."""
        import pandas as pd
        from services import player_stats_service

        logging.info("Engineering player features...")

        players_cols = ['{}_player_{}'.format(team, i) for team in ['home', 'away'] for i in range(1, 12)]

        player_stats_dict_series = self.raw_match_details.apply(
            lambda row: player_stats_service.get_player_stat(
                match_row=row,
                df_matches=self.raw_match_details,
                df_player_attr=self.raw_player_attributes,
                players=players_cols
            ),
            axis=1
        )

        new_player_stats_df = pd.json_normalize(player_stats_dict_series)
        self.match_and_player_details = pd.merge(self.raw_match_details, new_player_stats_df, how='left', on='match_api_id')

        self.next(self.end)

    # @card
    # @step
    # def clean_data(self):
    #     """Clean the datasets by handling missing values and outliers."""
    #     import pandas as pd
    #     import numpy as np
    #
    #     logging.info("Cleaning datasets...")
    #
    #     self.match_details = self.raw_match_details.copy()
    #     self.player_attributes = self.raw_player_attributes.copy()
    #
    #     self.match_details["date"] = pd.to_datetime(self.match_details["date"])
    #     self.player_attributes["date"] = pd.to_datetime(self.player_attributes["date"])
    #
    #     numeric_cols = self.player_attributes.select_dtypes(include=['float64', 'int64']).columns
    #     for col in numeric_cols:
    #         self.player_attributes[col].fillna(self.player_attributes[col].median(), inplace=True)
    #
    #     categorical_cols = self.player_attributes.select_dtypes(include=['object']).columns
    #     for col in categorical_cols:
    #         self.player_attributes[col].fillna(self.player_attributes[col].mode()[0], inplace=True)
    #
    #     self.next(self.engineer_features)

    # @card
    # @step
    # def engineer_features(self):
    #     """Create new features from existing data."""
    #     import pandas as pd
    #     import numpy as np
    #
    #     logging.info("Engineering features...")
    #
    #     # Create temporal features from dates
    #     self.match_details["year"] = self.match_details["date"].dt.year
    #     self.match_details["month"] = self.match_details["date"].dt.month
    #     self.match_details["day_of_week"] = self.match_details["date"].dt.dayofweek
    #
    #     # Calculate goal difference
    #     self.match_details["goal_difference"] = self.match_details["home_team_goal"] - self.match_details[
    #         "away_team_goal"]
    #
    #     # Calculate win/loss/draw indicator
    #     self.match_details["match_outcome"] = self.match_details["result_match"].map(
    #         {1: "home_win", 0: "draw", 2: "away_win"})
    #
    #     # Create team performance metrics (rolling averages)
    #     # Example: Calculate rolling average goals for home and away teams
    #     self.match_details["rolling_avg_goals_home"] = self.match_details.groupby("home_team")[
    #         "home_team_goal"].transform(
    #         lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    #     )
    #
    #     self.match_details["rolling_avg_goals_away"] = self.match_details.groupby("away_team")[
    #         "away_team_goal"].transform(
    #         lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    #     )
    #
    #     # Calculate scoring efficiency (goals per shot)
    #     self.match_details["shot_efficiency_ratio_home"] = self.match_details.apply(
    #         lambda row: row["home_team_goal"] / row["home_shoton"] if row["home_shoton"] > 0 else 0,
    #         axis=1
    #     )
    #
    #     # Create player-based aggregated features
    #     # For example, calculate average player ratings for each team per match
    #
    #     # Continue to feature selection and scaling
    #     self.next(self.scale_and_select_features)
    #
    # @card
    # @step
    # def scale_and_select_features(self):
    #     """Scale numerical features and select important ones."""
    #     from sklearn.preprocessing import StandardScaler, OneHotEncoder
    #     from sklearn.compose import ColumnTransformer
    #     from sklearn.pipeline import Pipeline
    #     from sklearn.impute import SimpleImputer
    #     import numpy as np
    #     import pandas as pd
    #
    #     logging.info("Scaling and selecting features...")
    #
    #     # Define feature groups
    #     numeric_features = [
    #         "home_team_goal", "away_team_goal",
    #         "rolling_avg_goals_home", "rolling_avg_goals_away",
    #         "shot_efficiency_ratio_home"
    #     ]
    #
    #     categorical_features = ["season", "match_outcome"]
    #
    #     # Create preprocessing pipelines
    #     numeric_transformer = Pipeline(steps=[
    #         ('imputer', SimpleImputer(strategy='median')),
    #         ('scaler', StandardScaler())
    #     ])
    #
    #     categorical_transformer = Pipeline(steps=[
    #         ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    #         ('onehot', OneHotEncoder(handle_unknown='ignore'))
    #     ])
    #
    #     # Create column transformer
    #     self.preprocessor = ColumnTransformer(
    #         transformers=[
    #             ('num', numeric_transformer, numeric_features),
    #             ('cat', categorical_transformer, categorical_features)
    #         ]
    #     )
    #
    #     # Fit and transform
    #     match_features = self.preprocessor.fit_transform(self.match_details)
    #
    #     # Convert to dataframe for easier handling
    #     # This assumes the transformed data is a dense array
    #     feature_names = (
    #             numeric_features +
    #             self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
    #                 categorical_features).tolist()
    #     )
    #
    #     self.processed_match_features = pd.DataFrame(
    #         match_features,
    #         columns=feature_names
    #     )
    #
    #     # Save original indices or match IDs to link back to original data
    #     self.processed_match_features["match_api_id"] = self.match_details["match_api_id"].values
    #
    #     # Continue to final data preparation
    #     self.next(self.prepare_final_datasets)
    #
    # @card
    # @step
    # def prepare_final_datasets(self):
    #     """Prepare final training, validation and test datasets."""
    #     from sklearn.model_selection import train_test_split
    #     import pandas as pd
    #
    #     logging.info("Preparing final datasets...")
    #
    #     # Sort by date to prevent data leakage
    #     self.match_details = self.match_details.sort_values("date")
    #
    #     # Split by time (assuming chronological importance)
    #     # Use the last 10% of matches as test data, previous 10% as validation
    #     test_size = int(len(self.match_details) * 0.1)
    #     val_size = int(len(self.match_details) * 0.1)
    #
    #     self.train_data = self.processed_match_features.iloc[:-test_size - val_size]
    #     self.val_data = self.processed_match_features.iloc[-test_size - val_size:-test_size]
    #     self.test_data = self.processed_match_features.iloc[-test_size:]
    #
    #     # Save the preprocessor and processed datasets
    #     self.next(self.save_artifacts)
    #
    # @card
    # @step
    # def save_artifacts(self):
    #     """Save all artifacts needed for model training."""
    #     import joblib
    #     import pandas as pd
    #     import os
    #
    #     logging.info("Saving artifacts...")
    #
    #     # Create output directories if they don't exist
    #     os.makedirs("../data/processed", exist_ok=True)
    #     os.makedirs("../models/preprocessors", exist_ok=True)
    #
    #     # Save preprocessed data
    #     self.train_data.to_csv("../data/processed/train_data.csv", index=False)
    #     self.val_data.to_csv("../data/processed/val_data.csv", index=False)
    #     self.test_data.to_csv("../data/processed/test_data.csv", index=False)
    #
    #     # Save preprocessor
    #     joblib.dump(self.preprocessor, "../models/preprocessors/match_preprocessor.joblib")
    #
    #     # Save feature metadata
    #     feature_metadata = {
    #         "numeric_features": self.train_data.select_dtypes(include=['float64', 'int64']).columns.tolist(),
    #         "categorical_features": self.train_data.select_dtypes(include=['object']).columns.tolist(),
    #         "train_shape": self.train_data.shape,
    #         "val_shape": self.val_data.shape,
    #         "test_shape": self.test_data.shape
    #     }
    #
    #     pd.DataFrame([feature_metadata]).to_json("../data/processed/feature_metadata.json", orient="records")
    #
    #     # End the pipeline
    #     self.next(self.end)

    @step
    def end(self):
        """End the Feature pipeline."""
        logging.info("Feature pipeline completed successfully!")

    def load_match_details(self):
        """Load match details dataset."""
        import pandas as pd
        return pd.read_csv('../data/raw/match_details.csv')

    def load_player_attributes(self):
        """Load player attributes dataset."""
        import pandas as pd
        return pd.read_csv('../data/raw/player_attributes.csv')


if __name__ == "__main__":
    FeaturePipeline()
