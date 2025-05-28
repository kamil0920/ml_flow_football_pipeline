import logging
import os
from pathlib import Path

import mlflow
import numpy as np

from common import (
    PYTHON,
    FlowMixin,
    build_features_transformer,
    build_target_transformer,
    configure_logging,
    packages,
    build_model,
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
        "xgboost",
        "boto3",
        "mlflow",
        "python-dotenv",
    ),
)
class Training(FlowSpec, FlowMixin):
    """Training pipeline using XGBoost for match result classification."""

    # Hyperparameters
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
    early_stopping_rounds = Parameter(
        "early-stopping-rounds", default=75,
        help="Number of rounds to perform early stopping"
    )
    f1_threshold = Parameter(
        "f1-threshold", default=0.55,
        help="Minimum CV f1 to register model"
    )
    temporal_validation_enabled = Parameter(
        "temporal-validation", default=True,
        help="Enable temporal validation splits"
    )
    n_older_seasons = Parameter(
        "n-older-seasons", default=7,
        help="Number of older seasons to include in training"
    )
    test_stage = Parameter(
        "test-stage", default=7,
        help="Stage to use as test set"
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
        """Initialize MLflow and parameters."""
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        logging.info("MLFLOW_TRACKING_URI: %s", self.mlflow_tracking_uri)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment("football_pipeline")
        self.mode = "production" if current.is_production else "development"
        logging.info("Running in %s mode", self.mode)

        run = mlflow.start_run(run_name=current.run_id)
        self.mlflow_run_id = run.info.run_id

        self.xgb_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'use_label_encoder': False,
            'early_stopping_rounds': self.early_stopping_rounds,
            'objective': 'binary:logistic'
        }
        mlflow.log_params(self.xgb_params)

        (
            self.X_train_raw,
            self.X_val_raw,
            self.X_test_raw,
            self.y_train_raw,
            self.y_val_raw,
            self.y_test_raw
        ) = self.load_and_split_data()

        self.next(self.prepare_temporal, self.transform_final)

    @step
    def prepare_temporal(self):
        self.temporal_splits = self.create_temporal_splits(max_test_stage=self.test_stage)
        self.next(self.train_temporal_splits, foreach='temporal_splits')

    @card
    @resources(memory=4096)
    @step
    def train_temporal_splits(self):
        split = self.input
        self.split_id = split['split_id']
        logging.info("Temporal split %d (test_stage=%d)", self.split_id, split['test_stage'])

        tgt = build_target_transformer()
        feat = build_features_transformer()

        y_tr = tgt.fit_transform(split['y_train'].values.reshape(-1, 1)).ravel()
        X_tr = feat.fit_transform(split['X_train'])
        y_val = tgt.transform(split['y_val'].values.reshape(-1, 1)).ravel()
        X_val = feat.transform(split['X_val'])
        y_tst = tgt.transform(split['y_test'].values.reshape(-1, 1)).ravel()
        X_tst = feat.transform(split['X_test'])

        print("Training labels:", np.unique(y_tr, return_counts=True))
        print("Validation labels:", np.unique(y_val, return_counts=True))

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id), \
                mlflow.start_run(run_name=f"temporal-{self.split_id}", nested=True) as run:
            self.mlflow_split_run_id = run.info.run_id
            mlflow.autolog(log_models=False)
            model = build_model(y_tr, **self.xgb_params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

            from sklearn.metrics import f1_score, log_loss
            preds = model.predict(X_tst)
            proba = model.predict_proba(X_tst)

            self.f1_score = f1_score(y_tst, preds)
            self.logloss = log_loss(y_tst, proba)
            mlflow.log_metrics({
                'val_f1': self.f1_score,
                'val_loss': self.logloss
            })

        self.next(self.aggregate_temporal)

    @step
    def aggregate_temporal(self, inputs):
        import numpy as np
        self.merge_artifacts(inputs, include=["mlflow_run_id", "mlflow_tracking_uri"])
        metrics = [(i.f1_score, i.logloss) for i in inputs]
        self.cv_f1, self.loss = np.mean(metrics, axis=0)
        self.cv_f1_std, self.loss_std = np.std(metrics, axis=0)

        logging.info("Accuracy: %f ±%f", self.cv_f1, self.loss)
        logging.info("Loss: %f ±%f", self.cv_f1_std, self.loss_std)

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_metrics({
                'temporal_f1': self.cv_f1,
                'temporal_f1_std': self.cv_f1_std,
                'temporal_loss': self.loss,
                'temporal_loss_std': self.loss_std
            })

        self.next(self.register)

    # @step
    # def transform_simple(self):
    #     """Transform the original simple train/val split."""
    #     tgt = build_target_transformer()
    #     feat = build_features_transformer()
    #     self.y_train_simple = tgt.fit_transform(
    #         self.y_train_raw.values.reshape(-1, 1)
    #     ).ravel()
    #     self.X_train_simple = feat.fit_transform(self.X_train_raw)
    #
    #     self.y_val_simple = tgt.transform(
    #         self.y_val_raw.values.reshape(-1, 1)
    #     ).ravel()
    #     self.X_val_simple = feat.transform(self.X_val_raw)
    #
    #     self.target_transformer = tgt
    #     self.features_transformer = feat
    #
    #     self.next(self.train_simple)

    # @step
    # def train_simple(self):
    #     import mlflow
    #     from sklearn.metrics import f1_score, log_loss
    #     mlflow.set_tracking_uri(self.mlflow_tracking_uri)
    #     with mlflow.start_run(run_id=self.mlflow_run_id):
    #         mlflow.autolog(log_models=False)
    #         model = build_model(self.y_train_simple, **self.xgb_params)
    #         model.fit(self.X_train_simple, self.y_train_simple)
    #
    #         preds = model.predict(self.X_val_simple)
    #         proba = model.predict_proba(self.X_val_simple)
    #         self.simple_f1 = f1_score(self.y_val_simple, preds)
    #         self.simple_loss = log_loss(self.y_val_simple, proba)
    #         mlflow.log_metrics({
    #             'simple_f1': self.simple_f1,
    #             'simple_loss': self.simple_loss
    #         })
    #
    #     import pandas as pd
    #     self.X_train_raw = pd.concat([self.X_train_raw, self.X_val_raw], ignore_index=True)
    #     self.y_train_raw = pd.concat([self.y_train_raw, self.y_val_raw], ignore_index=True)
    #     self.is_temporal = False
    #     self.next(self.join_branches)

    # @step
    # def join_branches(self, inputs):
    #     self.merge_artifacts(inputs, include=["mlflow_run_id", "mlflow_tracking_uri", "xgb_params"])
    #     chosen = next(i for i in inputs if i.is_temporal == self.temporal_validation_enabled)
    #     self.X_train_raw = chosen.X_train_raw
    #     self.y_train_raw = chosen.y_train_raw
    #     self.cv_f1 = chosen.cv_f1 if chosen.is_temporal else chosen.simple_f1
    #     self.loss = chosen.loss if chosen.is_temporal else chosen.simple_loss
    #     self.next(self.transform_final)

    @step
    def transform_final(self):
        tgt = build_target_transformer()
        feat = build_features_transformer()
        self.y_trn_final = tgt.fit_transform(self.y_train_raw.values.reshape(-1, 1)).ravel()
        self.X_trn_final = feat.fit_transform(self.X_train_raw)

        self.y_val_final = tgt.fit_transform(self.y_train_raw.values.reshape(-1, 1)).ravel()
        self.X_val_final = feat.fit_transform(self.X_train_raw)
        self.next(self.train_final)

    @card
    @resources(memory=4096)
    @step
    def train_final(self):
        import mlflow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.autolog(log_models=False)
            model = build_model(self.y_trn_final, **self.xgb_params)
            model.fit(self.X_trn_final, self.y_trn_final, eval_set=[(self.X_val_final, self.y_val_final)])
            mlflow.log_params(self.xgb_params)
        self.final_model = model
        self.next(self.register)

    @step
    def register(self, inputs):
        import tempfile
        import mlflow
        self.merge_artifacts(inputs)

        if self.cv_f1 >= self.f1_threshold:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            with (
                mlflow.start_run(run_id=self.mlflow_run_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                import mlflow.xgboost
                mlflow.xgboost.log_model(
                    self.final_model,
                    artifact_path='model',
                    registered_model_name='football',
                    code_paths=[(Path(__file__).parent / "inference.py").as_posix()],
                    artifacts=self._get_model_artifacts(directory),
                    pip_requirements=self._get_model_pip_requirements(),
                    signature=self._get_model_signature(),
                    example_no_conversion=True,
                )
                logging.info("Registered model with f1=%.3f", self.cv_f1)
        else:
            logging.info("Skipped registration: f1=%.3f below %.3f", self.cv_f1, self.f1_threshold)
        self.next(self.end)

    @step
    def end(self):
        logging.info("Flow complete!")


if __name__ == "__main__":
    Training()
