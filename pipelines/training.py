import logging
import os

import mlflow

from common import (
    PYTHON,
    FlowMixin,
    build_features_transformer,
    build_target_transformer,
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
        "xgboost",
        "boto3",
        "mlflow",
        "python-dotenv",
    ),
)
class Training(FlowSpec, FlowMixin):
    """Training pipeline using XGBoost for match result classification."""

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
    f1_threshold = Parameter(
        "f1-threshold", default=0.65,
        help="Minimum CV f1 to register model"
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
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        self.mode = "production" if current.is_production else "development"
        logging.info("Running in %s mode", self.mode)

        self.data = self.load_train_dataset()

        run = mlflow.start_run(run_name=current.run_id)
        self.mlflow_run_id = run.info.run_id

        self.xgb_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'use_label_encoder': False,
            'eval_metric': 'mlogloss'
        }
        mlflow.log_params(self.xgb_params)

        self.next(self.cross_validation, self.transform)

    @step
    def cross_validation(self):
        """Prepare 5-fold splits."""
        from sklearn.model_selection import KFold
        self.mlflow_tracking_uri = getattr(self, "mlflow_tracking_uri", "http://127.0.0.1:5000")
        self.mlflow_run_id = getattr(self, "mlflow_run_id", None)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        self.folds = list(enumerate(kf.split(self.data)))
        self.next(self.transform_fold, foreach='folds')

    @step
    def transform_fold(self):
        """Apply transformers on train/test split."""
        self.mlflow_tracking_uri = getattr(self, "mlflow_tracking_uri", "http://127.0.0.1:5000")
        self.mlflow_run_id = getattr(self, "mlflow_run_id", None)
        self.fold, (train_idx, test_idx) = self.input
        df = self.data

        y = df['result_match'].values
        df.drop('result_match', axis=1, inplace=True)

        y_train_reshape = y[train_idx].reshape(-1, 1)
        y_test_reshape = y[test_idx].reshape(-1,1)

        tgt = build_target_transformer()
        self.y_train = tgt.fit_transform(y_train_reshape).ravel()
        self.y_test = tgt.transform(y_test_reshape).ravel()
        self.target_transformer = tgt

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
        from sklearn.metrics import f1_score, log_loss
        self.mlflow_tracking_uri = getattr(self, "mlflow_tracking_uri", "http://127.0.0.1:5000")
        self.mlflow_run_id = getattr(self, "mlflow_run_id", None)
        self.run_id = getattr(self, "run_id", None)
        preds = self.model.predict(self.x_test)
        proba = self.model.predict_proba(self.x_test)
        self.f1_score = f1_score(self.y_test, preds)
        self.logloss = log_loss(self.y_test, proba)

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.start_run(run_id=self.run_id)
        mlflow.log_metrics({
            'f1_score': self.f1_score,
            'log_loss': self.logloss
        })
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self, inputs):
        """Aggregate CV metrics."""
        import numpy as np
        self.mlflow_tracking_uri = getattr(inputs[0], "mlflow_tracking_uri", "http://127.0.0.1:5000")
        self.mlflow_run_id = getattr(inputs[0], "mlflow_run_id", None)
        self.metrics = [ (i.f1_score, i.logloss) for i in inputs ]
        f1, losses = zip(*self.metrics)
        self.cv_f1 = np.mean(f1)
        self.cv_f1_std = np.std(f1)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.start_run(run_id=self.mlflow_run_id)
        mlflow.log_metrics({
            'cv_f1': self.cv_f1,
            'cv_f1_std': self.cv_f1_std
        })
        self.next(self.register_model)

    @step
    def transform(self):
        """Fit transformers on full data."""
        self.mlflow_tracking_uri = getattr(self, "mlflow_tracking_uri", "http://127.0.0.1:5000")
        self.mlflow_run_id = getattr(self, "mlflow_run_id", None)
        df = self.data
        y = df['result_match'].values
        df.drop('result_match', axis=1, inplace=True)

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
        import mlflow
        self.merge_artifacts(inputs)
        self.mlflow_tracking_uri = getattr(self, "mlflow_tracking_uri", "http://127.0.0.1:5000")
        self.mlflow_run_id = getattr(self, "mlflow_run_id", None)
        if self.cv_f1 >= self.f1_threshold:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            with mlflow.start_run(run_id=self.mlflow_run_id):
                import mlflow.xgboost
                mlflow.xgboost.log_model(
                    self.model,
                    artifact_path='model',
                    registered_model_name='football'
                )
        else:
            logging.info("CV f1 %.3f below threshold %.3f, skipping registration",
                         self.cv_f1, self.f1_threshold)
        self.next(self.end)

    @step
    def end(self):
        logging.info("Training flow completed.")

if __name__ == "__main__":
    Training()
