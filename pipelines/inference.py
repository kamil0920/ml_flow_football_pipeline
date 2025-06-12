import logging
import logging.config
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModelContext


class Model(mlflow.pyfunc.PythonModel):
    """A custom model that can be used to make predictions.

    This model implements an inference pipeline with three phases: preprocessing,
    prediction, and postprocessing. The model will optionally store the input requests
    and predictions in a SQLite database.

    The [Custom MLflow Models with mlflow.pyfunc](https://mlflow.org/blog/custom-pyfunc)
    blog post is a great reference to understand how to use custom Python models in
    MLflow.
    """

    def __init__(
            self,
            data_collection_uri: str | None = "football.db",
            *,
            data_capture: bool = False,
    ) -> None:
        """Initialize the model.

        By default, the model will not collect the input requests and predictions. This
        behavior can be overwritten on individual requests.

        This constructor expects the connection URI to the storage medium where the data
        will be collected. By default, the data will be stored in a SQLite database
        named "football" and located in the root directory from where the model runs.
        You can override the location by using the 'DATA_COLLECTION_URI' environment
        variable.
        """
        self.data_capture = data_capture
        self.data_collection_uri = data_collection_uri

    def load_context(self, context: PythonModelContext) -> None:
        """Load the transformers and the XGBoost model specified as artifacts.

        This function is called only once as soon as the model is constructed.
        """

        import xgboost

        self._configure_logging()
        logging.info("Loading model context...")

        self.data_collection_uri = os.environ.get(
            "DATA_COLLECTION_URI",
            self.data_collection_uri,
        )

        logging.info("Data collection URI: %s", self.data_collection_uri)

        self.features_transformer = joblib.load(context.artifacts["features_transformer"])
        self.target_transformer = joblib.load(context.artifacts["target_transformer"])
        self.model = joblib.load(context.artifacts["model"])

        logging.info("Model is ready to receive requests")

    def predict(
            self,
            context: PythonModelContext,  # noqa: ARG002
            model_input: pd.DataFrame | list[dict[str, Any]] | dict[str, Any] | list[Any],
            params: dict[str, Any] | None = None,
    ) -> list:
        """Handle the request received from the client.

        This method is responsible for processing the input data received from the
        client, making a prediction using the model, and returning a readable response
        to the client.

        The caller can specify whether we should capture the input request and
        prediction by using the `data_capture` parameter when making a request.
        """
        if isinstance(model_input, list):
            model_input = pd.DataFrame(model_input)

        if isinstance(model_input, dict):
            model_input = pd.DataFrame([model_input])

        logging.info(
            "Received prediction request with %d %s",
            len(model_input),
            "samples" if len(model_input) > 1 else "sample",
        )

        model_output = []

        transformed_payload = self.process_input(model_input)
        if transformed_payload is not None:
            logging.info("Making a prediction using the transformed payload...")
            predictions = self.model.predict_proba(transformed_payload)

            model_output = self.process_output(predictions)

        # If the caller specified the `data_capture` parameter when making the
        # request, we should use it to determine whether we should capture the
        # input request and prediction.
        if self.should_capture(params):
            self.capture(model_input, model_output)

        logging.info("Returning prediction to the client")
        logging.debug("%s", model_output)

        return model_output

    def should_capture(self, params: dict[str, Any] | None) -> bool:
        if params is None:
            return self.data_capture
        return bool(params.get("data_capture", False))

    def process_input(self, payload: pd.DataFrame) -> pd.DataFrame:
        """Process the input data received from the client.

        This method is responsible for transforming the input data received from the
        client into a format that can be used by the model.
        """
        logging.info("Transforming payload...")

        try:
            result = self.features_transformer.transform(payload)
        except Exception:
            logging.exception("There was an error processing the payload.")
            return None

        return result

    def process_output(self, proba):
        """Process the prediction received from the model.
        This method is responsible for transforming the prediction received from the
        model into a readable format that will be returned to the client.
        """
        idx = np.argmax(proba, axis=1)
        confidence = np.max(proba, axis=1)

        labels = (
            self.target_transformer
            .inverse_transform(idx.reshape(-1, 1))
            .ravel()
        )

        return [
            {"prediction": str(lbl), "confidence": float(conf)}
            for lbl, conf in zip(labels, confidence, strict=True)
        ]

    def capture(self, model_input: pd.DataFrame, model_output: list) -> None:
        """Save the input request and output prediction to the database.

        This method will save the input request and output prediction to a SQLite
        database. If the database doesn't exist, this function will create it.
        """
        logging.info("Storing input payload and predictions in the database...")

        connection = None
        try:
            connection = sqlite3.connect(self.data_collection_uri)

            # Let's create a copy from the model input so we can modify the DataFrame
            # before storing it in the database.
            data = model_input.copy()

            # We need to add the current time, the prediction and confidence columns
            # to the DataFrame to store everything together.
            data["date"] = datetime.now(timezone.utc)

            # Let's initialize the prediction and confidence columns with None. We'll
            # overwrite them later if the model output is not empty.
            data["prediction"] = None
            data["confidence"] = None

            # Let's also add a column to store the ground truth. This column can be
            # used by the labeling team to provide the actual result_match for the data.
            data["result_match"] = None

            # If the model output is not empty, we should update the prediction and
            # confidence columns with the corresponding values.
            if model_output is not None and len(model_output) > 0:
                data["prediction"] = [item["prediction"] for item in model_output]
                data["confidence"] = [item["confidence"] for item in model_output]

            # Let's automatically generate a unique identified for each row in the
            # DataFrame. This will be helpful later when labeling the data.
            data["uuid"] = [str(uuid.uuid4()) for _ in range(len(data))]

            # Finally, we can save the data to the database.
            data.to_sql("data", connection, if_exists="append", index=False)

        except sqlite3.Error:
            logging.exception(
                "There was an error saving the input request and output prediction "
                "in the database.",
            )
        finally:
            if connection:
                connection.close()

    def _configure_logging(self):
        """Configure how the logging system will behave."""
        import sys
        from pathlib import Path

        if Path("logging.conf").exists():
            logging.config.fileConfig("logging.conf")
        else:
            logging.basicConfig(
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[logging.StreamHandler(sys.stdout)],
                level=logging.INFO,
            )
