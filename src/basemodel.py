from typing import Dict, Union, Optional, Callable
import os

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import mlflow
from mlflow.models.signature import infer_signature

from src.preprocessing import transform_ts_data_into_features_and_target
from src.logger import get_console_logger

logger = get_console_logger()

def get_baseline_model_error(X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Returns the baseline model error."""
    predictions = X_test['price_1_hour_ago']
    return mean_absolute_error(y_test, predictions)

def train(X: pd.DataFrame, y: pd.Series, params: Optional[Dict[str, float]]=None) -> None:
    """
    Train a logistic regression model using the input features `X` and targets `y`,
    and log the experiment with MLflow.
    """
    # Set up MLflow tracking URI and experiment
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Dummy model")

    with mlflow.start_run():
        # Log the parameters
        #mlflow.log_params(params)

        # split the data into train and test
        train_sample_size = int(0.9 * len(X))
        X_train, X_test = X[:train_sample_size], X[train_sample_size:]
        y_train, y_test = y[:train_sample_size], y[train_sample_size:]
        logger.info(f'Train sample size: {len(X_train)}')
        logger.info(f'Test sample size: {len(X_test)}')


        # baseline model performance
        baseline_mae = get_baseline_model_error(X_test, y_test)
        logger.info(f'Test MAE: {baseline_mae}')
        mlflow.log_metric("Test_MAE", baseline_mae)

        # Set tags for the run
        mlflow.set_tag("Training Info", "Dummy Basemodel")

        # Infer model signature
        #signature = infer_signature(X_train, model.predict(X_train))

        # Log the model
        #mlflow.sklearn.log_model(
        #    sk_model=model,
        #    artifact_path="model_artifact",
        #    signature=signature,
        #    input_example=X_train.iloc[:5],
        #    registered_model_name="MLflow-test-Example"
        #)

if __name__ == '__main__':
    logger.info('Generating features and targets')
    # Assuming your data is already loaded as DataFrame 'features' and Series 'target'
    features, target = transform_ts_data_into_features_and_target()

    # Define parameters for the model
    logger.info('Starting training')
    train(features, target)
