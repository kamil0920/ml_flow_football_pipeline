import logging
import os
from pathlib import Path

import mlflow
import numpy as np
from matplotlib import pyplot as plt
from mlflow import pyfunc
from inference import Model
import xgboost as xgb

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
    n_older_seasons = Parameter(
        "n-older-seasons", default=7,
        help="Number of older seasons to include in training"
    )
    test_stage = Parameter(
        "test-stage", default=7,
        help="Stage to use as test set"
    )
    positive_class = Parameter(
        "positive-class", default='H',
        help="Class to use as positive class"
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

        tgt = build_target_transformer(positive=self.positive_class)
        feat = build_features_transformer()

        y_tr = tgt.fit_transform(split['y_train']).ravel()
        X_tr = feat.fit_transform(split['X_train'])
        y_val = tgt.transform(split['y_val']).ravel()
        X_val = feat.transform(split['X_val'])
        y_tst = tgt.transform(split['y_test']).ravel()
        X_tst = feat.transform(split['X_test'])

        feature_names = feat.get_feature_names_out()
        print(f'Feature names: {feature_names}')

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

    @step
    def transform_final(self):
        self.tgt = build_target_transformer(positive=self.positive_class)
        self.feat = build_features_transformer()
        self.y_trn_final = self.tgt.fit_transform(self.y_train_raw).ravel()
        self.X_trn_final = self.feat.fit_transform(self.X_train_raw)

        self.y_val_final = self.tgt.fit_transform(self.y_train_raw).ravel()
        self.X_val_final = self.feat.fit_transform(self.X_train_raw)

        self.feature_names = self.feat.get_feature_names_out().tolist()

        self.next(self.train_final)

    @card
    @resources(memory=4096)
    @step
    def train_final(self):
        import mlflow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.autolog(log_models=False)
            evals = [(self.X_val_final, self.y_val_final)]
            model = build_model(self.X_trn_final, **self.xgb_params)
            model.fit(self.X_trn_final, self.y_trn_final, eval_set=evals)
            mlflow.log_params(self.xgb_params)
            booster = model.get_booster()
            booster.feature_names = self.feature_names

            fig, ax = plt.subplots(figsize=(22, 14))
            xgb.plot_importance(booster, importance_type="weight", ax=ax)
            ax.set_title("Feature Importance")
            mlflow.log_figure(fig, "feature_importance.png")
            plt.close(fig)

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
                import mlflow.pyfunc
                pyfunc.log_model(
                    python_model=Model(data_capture=False),
                    artifact_path='model',
                    registered_model_name='football',
                    code_paths=[(Path(__file__).parent / "inference.py").as_posix(),
                                (Path(__file__).parent / "common.py").as_posix(),
                                ],
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

    def _get_model_artifacts(self, directory: str):
        """Return the list of artifacts that will be included with model.

        The model must preprocess the raw input data before making a prediction, so we
        need to include the Scikit-Learn transformers as part of the model package.
        """
        import joblib

        model_path = (Path(directory) / "model.joblib").as_posix()
        joblib.dump(self.final_model, model_path)

        features_transformer_path = (Path(directory) / "features.joblib").as_posix()
        target_transformer_path = (Path(directory) / "target.joblib").as_posix()
        joblib.dump(self.feat, features_transformer_path)
        joblib.dump(self.tgt, target_transformer_path)

        return {
            "model": model_path,
            "features_transformer": features_transformer_path,
            "target_transformer": target_transformer_path,
        }

    def _get_model_signature(self):
        """Return the model's signature.

        The signature defines the expected format for model inputs and outputs. This
        definition serves as a uniform interface for appropriate and accurate use of
        a model.
        """
        from mlflow.models import infer_signature

        return infer_signature(
            model_input=self.X_train.head(1),
            model_output={"prediction": "home_win", "confidence": 0.90},
            params={"data_capture": False},
        )

    def _get_model_pip_requirements(self):
        """Return the list of required packages to run the model in production."""
        return [
            f"{package}=={version}"
            for package, version in packages(
                "scikit-learn",
                "pandas",
                "numpy",
                "xgboost",
            ).items()
        ]


if __name__ == "__main__":
    Training()
