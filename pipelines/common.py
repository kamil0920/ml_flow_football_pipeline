import logging
import logging.config
import os
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
from metaflow import S3, IncludeFile, current

PYTHON = "3.12"

PACKAGES = {
    "scikit-learn": "1.5.2",
    "pandas": "2.2.3",
    "numpy": "2.1.1",
    "xgboost": "3.0.0",
    "boto3": "1.35.32",
    "packaging": "24.1",
    "mlflow": "2.17.1",
    "setuptools": "75.1.0",
    "requests": "2.32.3",
    "evidently": "0.4.33",
    "azure-ai-ml": "1.19.0",
    "azureml-mlflow": "1.57.0.post1",
    "python-dotenv": "1.0.1",
}

class FlowMixin:
    dataset = IncludeFile(
        "match_stats",
        is_text=True,
        help=(
            "Local copy of the match_stats dataset. This file will be included in the "
            "flow and will be used whenever the flow is executed in development mode."
        ),
        default="data/match_stats.csv",
    )

    def load_dataset(self):
        import numpy as np

        if current.is_production:
            dataset = os.environ.get("DATASET", self.dataset)

            with S3(s3root=dataset) as s3:
                files = s3.get_all()

                logging.info("Found %d file(s) in remote location", len(files))

                raw_data = [pd.read_csv(StringIO(file.text)) for file in files]
                data = pd.concat(raw_data)
        else:
            data = pd.read_csv(StringIO(self.dataset))


        seed = int(time.time() * 1000) if current.is_production else 42
        generator = np.random.default_rng(seed=seed)
        data = data.sample(frac=1, random_state=generator)

        logging.info("Loaded dataset with %d samples", len(data))

        return data


def packages(*names: str):
    return {name: PACKAGES[name] for name in names if name in PACKAGES}


def configure_logging():
    if Path("logging.conf").exists():
        logging.config.fileConfig("logging.conf")
    else:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            level=logging.INFO,
        )

def build_model(learning_rate=0.01):
    from xgboost import XGBClassifier

    model = XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=100,
        max_depth=3,
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric=['logloss', 'aucpr']
    )

    return model
