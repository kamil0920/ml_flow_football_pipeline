import logging
import os
from pathlib import Path

import pandas as pd
import mlflow
from common import (
    PYTHON,
    FlowMixin,
    build_features_transformer,
    build_target_transformer,
    configure_logging,
    packages,
)
from inference import Model
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


@project(name="penguins")
@pypi_base(
    python=PYTHON,
    packages=packages(
        "scikit-learn",
        "pandas",
        "numpy",
        "xgboost",
        "boto3",
        "mlflow",
        "python-dotenv",
    ),
)
class Training(FlowSpec, FlowMixin):
    """Training pipeline using XGBoost for match result classification."""

    # XGBoost hyperparameters
    n_estimators = Parameter(
        "n-estimators", default=100,
        help="Number of trees in the XGBoost ensemble"
    )
    max_depth = Parameter(
        "max-depth", default=6,
        help="Maximum tree depth for base learners"
    )
    learning_rate = Parameter(
        "learning-rate", default=0.1,
        help="Boosting learning rate"
    )
    subsample = Parameter(
        "subsample", default=0.8,
        help="Subsample ratio of the training instances"
    )
    accuracy_threshold = Parameter(
        "accuracy-threshold", default=0.7,
        help="Minimum CV accuracy to register model"
    )

    @card
    @environment(
        vars={
            "MLFLOW_TRACKING_URI": os.getenv(
                "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
            ),
        },
    )
    @step
    def start(self):
        """Initialize MLflow, load data, and set params."""
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        self.mode = "production" if current.is_production else "development"
        logging.info("Running in %s mode", self.mode)

        # load dataset
        self.data = self.load_dataset()

        # start MLflow run
        run = mlflow.start_run(run_name=current.run_id)
        self.mlflow_run_id = run.info.run_id

        # collect XGB params
        self.xgb_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'use_label_encoder': False,
            'eval_metric': 'mlogloss'
        }
        mlflow.log_params(self.xgb_params)

        # parallel: cv and full-train
        self.next(self.cross_validation, self.transform)

    @step
    def cross_validation(self):
        """Prepare 5-fold splits."""
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        self.folds = list(enumerate(kf.split(self.data)))
        self.next(self.transform_fold, foreach='folds')

    @step
    def transform_fold(self):
        """Apply transformers on train/test split."""
        self.fold, (train_idx, test_idx) = self.input
        df = self.data
        y = df['species'].values

        # target
        tgt = build_target_transformer()
        self.y_train = tgt.fit_transform(y[train_idx].reshape(-1,1)).ravel()
        self.y_test = tgt.transform(y[test_idx].reshape(-1,1)).ravel()
        self.target_transformer = tgt

        # features
        feat = build_features_transformer()
        self.x_train = feat.fit_transform(df.iloc[train_idx])
        self.x_test = feat.transform(df.iloc[test_idx])
        self.features_transformer = feat

        self.next(self.train_fold)

    @card
    @resources(memory=4096)
    @step
    def train_fold(self):
        """Train XGBClassifier on one fold."""
        import mlflow.xgboost
        from xgboost import XGBClassifier

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id, nested=True) as run:
            mlflow.xgboost.autolog()
            model = XGBClassifier(**self.xgb_params)
            model.fit(self.x_train, self.y_train)
            self.model = model
            self.run_id = run.info.run_id
        self.next(self.evaluate_fold)

    @step
    def evaluate_fold(self):
        """Evaluate on test split."""
        from sklearn.metrics import accuracy_score, log_loss
        preds = self.model.predict(self.x_test)
        proba = self.model.predict_proba(self.x_test)
        self.accuracy = accuracy_score(self.y_test, preds)
        self.logloss = log_loss(self.y_test, proba)

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.start_run(run_id=self.run_id)
        mlflow.log_metrics({
            'accuracy': self.accuracy,
            'log_loss': self.logloss
        })
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self, inputs):
        """Aggregate CV metrics."""
        import numpy as np
        self.metrics = [ (i.accuracy, i.logloss) for i in inputs ]
        accs, losses = zip(*self.metrics)
        self.cv_accuracy = np.mean(accs)
        self.cv_accuracy_std = np.std(accs)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.start_run(run_id=self.mlflow_run_id)
        mlflow.log_metrics({
            'cv_accuracy': self.cv_accuracy,
            'cv_accuracy_std': self.cv_accuracy_std
        })
        self.next(self.register_model)

    @step
    def transform(self):
        """Fit transformers on full data."""
        df = self.data
        y = df['species'].values
        self.target_transformer = build_target_transformer()
        self.y = self.target_transformer.fit_transform(y.reshape(-1,1)).ravel()
        self.features_transformer = build_features_transformer()
        self.x = self.features_transformer.fit_transform(df)
        self.next(self.train_model)

    @card
    @resources(memory=4096)
    @step
    def train_model(self):
        """Train final XGBClassifier on all data."""
        import mlflow.xgboost
        from xgboost import XGBClassifier

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.xgboost.autolog()
            model = XGBClassifier(**self.xgb_params)
            model.fit(self.x, self.y)
            self.model = model
        self.next(self.register_model)

    @step
    def register_model(self, inputs):
        """Register if CV acc >= threshold."""
        self.merge_artifacts(inputs)
        if self.cv_accuracy >= self.accuracy_threshold:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            with mlflow.start_run(run_id=self.mlflow_run_id):
                mlflow.xgboost.log_model(
                    self.model,
                    artifact_path='model',
                    registered_model_name='penguins'
                )
        else:
            logging.info("CV accuracy %.3f below threshold %.3f, skipping registration",
                         self.cv_accuracy, self.accuracy_threshold)
        self.next(self.end)

    @step
    def end(self):
        logging.info("Training flow completed.")

    def load_dataset(self):
        return pd.read_parquet('../preprocessed/df.csv')


if __name__ == "__main__":
    Training()
