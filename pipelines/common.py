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
            data = pd.read_csv(StringIO(self.train_data_path))

        seed = int(time.time() * 1000) if current.is_production else 42
        rng = np.random.default_rng(seed)
        data = data.sample(frac=1, random_state=rng).reset_index(drop=True)

        logging.info("Loaded dataset with %d rows", len(data))
        return data

class OutlierHandler(BaseEstimator, TransformerMixin):
    """Detects outliers with IsolationForest and masks them to NaN."""
    def __init__(self, contamination=0.05, random_state=42):
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X, y=None):
        self.iforest_ = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.iforest_.fit(X)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=getattr(self, 'feature_names_in_', None) or range(X.shape[1]))
        mask = self.iforest_.predict(X) == 1
        X_clean = X.where(mask, np.nan)
        return X_clean.values

def build_target_transformer():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    """
    Returns a 1-step pipeline that maps result_match 'H'→1, others→0.
    """
    def map_result(df: pd.DataFrame) -> pd.DataFrame:
        print(f'before: {df.result_match.value_counts()}')
        frame = (df['result_match'] == 'H').astype(int).to_frame()

        print(f'after: {frame.result_match.value_counts()}')

        return frame

    return Pipeline([
        ('map_result', FunctionTransformer(map_result, validate=False))
    ])


def build_features_transformer():
    """Preprocess numeric and categorical features."""
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    numeric_pipe = make_pipeline(
        SimpleImputer(strategy="mean"),
        OutlierHandler(contamination=0.05),
        SimpleImputer(strategy="median"),
    )
    categorical_pipe = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )
    return ColumnTransformer([
        ("num", numeric_pipe, make_column_selector(dtype_exclude="object")),
        ("cat", categorical_pipe, ["seasonal_context", "day_of_week", "is_weekend"]),
    ])


def build_model(**xgb_params):
    """Instantiate an XGBClassifier with provided hyperparameters."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        **xgb_params,
        use_label_encoder=False,
        eval_metric=["logloss", "aucpr"],
    )
