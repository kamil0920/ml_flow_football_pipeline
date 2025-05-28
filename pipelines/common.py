import logging
import logging.config
import os
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from metaflow import S3, Parameter, current
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import IsolationForest

default_py = "3.12"

PACKAGES = {
    "scikit-learn": "1.5.2",
    "pandas": "2.2.3",
    "numpy": "2.1.1",
    "xgboost": "3.0.0",
    "boto3": "1.35.32",
    "mlflow": "2.17.1",
    "python-dotenv": "1.0.1",
}

PYTHON = default_py


def packages(*names: str):
    """Helper to pick only required packages and their pinned versions."""
    return {name: PACKAGES[name] for name in names if name in PACKAGES}


def configure_logging():
    """Set up basic logging or from file if present."""
    if Path("logging.conf").exists():
        logging.config.fileConfig("logging.conf")
    else:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            level=logging.INFO,
        )


class FlowMixin:
    """Shared code for loading datasets in dev vs prod."""
    match_details_dataset = Parameter(
        "match-details-dataset",
        help="Local copy of the match_stats dataset. This file will be included in the "
             "flow and will be used whenever the flow is executed in development mode."
        ,
        default="data/raw/match_details.csv",
    )
    players_stats_dataset = Parameter(
        "players-stats-dataset",
        help="Local copy of the players_stats_dataset dataset. This file will be included in the "
             "flow and will be used whenever the flow is executed in development mode."
        ,
        default="data/raw/player_attributes.csv",
    )

    def load_raw_match_details_dataset(self):
        import numpy as np

        if current.is_production:
            match_details_s3_root = os.environ.get("MATCH_DATASET", None)
            player_stats_s3_root = os.environ.get("PLAYER_DATASET", None)
            with S3(s3root=match_details_s3_root) as s3:
                files = s3.get_all()
                logging.info("Found %d remote file(s)", len(files))
                frames = [pd.read_csv(StringIO(f.text)) for f in files]
                match_data = pd.concat(frames, ignore_index=True)
            with S3(s3root=player_stats_s3_root) as s3:
                files = s3.get_all()
                logging.info("Found %d remote file(s)", len(files))
                frames = [pd.read_csv(StringIO(f.text)) for f in files]
                player_data = pd.concat(frames, ignore_index=True)
        else:
            match_data = pd.read_csv(self.match_details_dataset)
            player_data = pd.read_csv(self.players_stats_dataset)

        seed = int(time.time() * 1000) if current.is_production else 42
        rng = np.random.default_rng(seed)
        match_data = match_data.sample(frac=1, random_state=rng).reset_index(drop=True)
        player_data = player_data.sample(frac=1, random_state=rng).reset_index(drop=True)

        logging.info("Loaded match details dataset with %d rows", len(match_data))
        logging.info("Loaded player stats dataset with %d rows", len(player_data))
        return match_data, player_data

    train_data_path = Parameter(
        "train-data",
        help="Local path or S3 prefix for your preprocessed train_data.csv",
        default="data/preprocessed/train_data.csv",
    )

    def load_train_dataset(self):
        import numpy as np

        if current.is_production:
            s3_root = os.environ.get("DATASET", None)
            with S3(s3root=s3_root) as s3:
                files = s3.get_all()
                logging.info("Found %d remote file(s)", len(files))
                frames = [pd.read_csv(StringIO(f.text)) for f in files]
                data = pd.concat(frames, ignore_index=True)
        else:
            data = pd.read_csv(self.train_data_path)

        seed = int(time.time() * 1000) if current.is_production else 42
        rng = np.random.default_rng(seed)
        data = data.sample(frac=1, random_state=rng).reset_index(drop=True)

        logging.info("Loaded dataset with %d rows", len(data))
        return data

    n_older_seasons = Parameter(
        "n-older-seasons",
        default=7,
        type=int,
        help="Number of older seasons to include in training"
    )

    def create_temporal_splits(self, max_test_stage=None):
        """Create multiple temporal splits for validation."""
        import pandas as pd

        df_matches = self.load_train_dataset()
        df_matches = df_matches.sort_values(by=["season", "stage", "date"])

        sorted_seasons = sorted(df_matches["season"].unique())
        newest_season = sorted_seasons[-1]
        older_seasons = sorted_seasons[:-1]

        if max_test_stage is None:
            max_test_stage = df_matches.loc[df_matches["season"] == newest_season, "stage"].max()

        temporal_splits = []

        for test_stage in range(4, max_test_stage + 1):  # Changed from 3 to 4
            # Use two stages for validation
            val_stages = [test_stage - 2, test_stage - 1]
            min_val_stage = min(val_stages)

            # Skip if we don't have enough stages for validation
            if min_val_stage < 2:
                continue

            train_seasons = sorted(older_seasons[-self.n_older_seasons:], reverse=True)

            # Training data: older seasons + newest season before validation stages
            X_train_old = df_matches[df_matches["season"].isin(train_seasons)]
            X_train_new = df_matches[
                (df_matches["season"] == newest_season) &
                (df_matches["stage"] < min_val_stage)  # Before the first validation stage
                ]
            df_train = pd.concat([X_train_old, X_train_new], ignore_index=True)

            # Validation data: two consecutive stages before test stage
            df_val = df_matches[
                (df_matches["season"] == newest_season) &
                (df_matches["stage"].isin(val_stages))  # Two stages for validation
                ].reset_index(drop=True)

            # Test data: remains the same
            df_test = df_matches[
                (df_matches["season"] == newest_season) &
                (df_matches["stage"] == test_stage)
                ].reset_index(drop=True)

            if len(df_train) == 0 or len(df_val) == 0 or len(df_test) == 0:
                continue

            feature_cols_to_drop = [
                'match_api_id',
                'season',
                'stage',
                'date',
                'result_match',
                'points_home',
                'points_away'
            ]

            split_data = {
                'split_id': len(temporal_splits),
                'test_stage': test_stage,
                'val_stages': val_stages,  # Changed from val_stage to val_stages
                'X_train': df_train.drop(columns=feature_cols_to_drop),
                'y_train': df_train["result_match"],
                'X_val': df_val.drop(columns=feature_cols_to_drop),
                'y_val': df_val["result_match"],
                'X_test': df_test.drop(columns=feature_cols_to_drop),
                'y_test': df_test["result_match"],
                'train_size': len(df_train),
                'val_size': len(df_val),
                'test_size': len(df_test)
            }

            temporal_splits.append(split_data)

        logging.info(f"Created {len(temporal_splits)} temporal validation splits")
        return temporal_splits

    def load_and_split_data(self, test_size=0.15, val_size=0.15, random_state=None):
        """Load dataset and create train/val/test splits for simple validation."""
        from sklearn.model_selection import train_test_split

        df = self.load_train_dataset()

        feature_cols_to_drop = [
            'match_api_id',
            'season',
            'stage',
            'date',
            'result_match',
            'points_home',
            'points_away'
        ]

        X = df.drop(columns=feature_cols_to_drop)
        y = df["result_match"]

        if random_state is None:
            random_state = 42 if not current.is_production else int(time.time() * 1000) % 2 ** 32

        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )

        logging.info("Data split completed:")
        logging.info(f"  Train: {len(self.X_train)} samples")
        logging.info(f"  Validation: {len(self.X_val)} samples")
        logging.info(f"  Test: {len(self.X_test)} samples")

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Detects outliers using the IQR method and masks them to NaN, only on numeric columns.
    Non-numeric columns are passed through unchanged.
    """

    def __init__(self, iqr_multiplier=1.5):
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=getattr(self, 'feature_names_in_', None)
                                        or range(X.shape[1]))

        numeric = X.select_dtypes(include=[np.number])
        self.q1_ = numeric.quantile(0.1, numeric_only=True)
        self.q3_ = numeric.quantile(0.9, numeric_only=True)
        self.iqr_ = self.q3_ - self.q1_

        self.lower_bound_ = self.q1_ - self.iqr_multiplier * self.iqr_
        self.upper_bound_ = self.q3_ + self.iqr_multiplier * self.iqr_

        self.numeric_cols_ = numeric.columns.tolist()
        self.all_cols_ = X.columns.tolist()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.all_cols_)

        X_out = X.copy()

        numeric = X_out[self.numeric_cols_]
        mask = (numeric >= self.lower_bound_) & (numeric <= self.upper_bound_)
        X_out[self.numeric_cols_] = numeric.where(mask, np.nan)

        return X_out.values


def map_result(df: pd.DataFrame) -> pd.DataFrame:
    frame = (df == 'H').astype(int)
    return frame


def build_target_transformer():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    """
    Returns a 1-step pipeline that maps result_match 'H'→1, others→0.
    """
    return Pipeline([
        ('map_result', FunctionTransformer(map_result, validate=False))
    ])


def build_features_transformer():
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    numeric_pipe = make_pipeline(
        OutlierHandler(),
        SimpleImputer(strategy="median"),
    )
    categorical_pipe = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )
    return ColumnTransformer([
        ("num", numeric_pipe, make_column_selector(dtype_exclude="object")),
        ("cat", categorical_pipe, ["seasonal_context", "day_of_week", "is_weekend"]),
    ])


def build_model(y_train, **xgb_params):
    """Instantiate an XGBClassifier with provided hyperparameters."""
    from xgboost import XGBClassifier

    xgb_params.setdefault("use_label_encoder", False)
    xgb_params.setdefault("eval_metric", ["logloss", "aucpr"])
    unique, counts = np.unique(y_train, return_counts=True)
    count_dict = dict(zip(unique, counts))
    ratio = count_dict[0] / count_dict[1]

    xgb_params["scale_pos_weight"] = ratio

    return XGBClassifier(**xgb_params)
