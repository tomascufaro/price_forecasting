from typing import Dict, Union, Optional, Callable
import os
from pathlib import Path

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pickle

import mlflow
from mlflow import log_metric, log_param, log_artifacts
from mlflow.sklearn import log_model

from src.preprocessing import (
    transform_ts_data_into_features_and_target,
    get_preprocessing_pipeline
)
from src.hyperparams import find_best_hyperparams
from src.logger import get_console_logger
from src.path import MODELS_DIR

logger = get_console_logger()

def get_baseline_model_error(X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Returns the baseline model error."""
    predictions = X_test['price_1_hour_ago']
    return mean_absolute_error(y_test, predictions)

def get_model_fn_from_name(model_name: str) -> Callable:
    """Returns the model function given the model name."""
    if model_name == 'lasso':
        return Lasso
    elif model_name == 'xgboost':
        return XGBRegressor
    elif model_name == 'lightgbm':
        return LGBMRegressor
    else:
        raise ValueError(f'Unknown model name: {model_name}')

def train(
    X: pd.DataFrame,
    y: pd.Series,
    model: str,
    tune_hyperparams: Optional[bool] = False,
    hyperparam_trials: Optional[int] = 10,
    ) -> None:
    """
    Train a model using the input features `X` and targets `y`,
    possibly running hyperparameter tuning.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Adjust your MLflow URI
    mlflow.set_experiment("Model Training")
    
    with mlflow.start_run():
        model_fn = get_model_fn_from_name(model)
        mlflow.set_tag("model", model)
            # Log the features dataframe as an artifact
        features_path = "features.csv"
        features.to_csv(features_path, index=False)
        mlflow.log_artifact(features_path, artifact_path="features")

        # Log the target dataframe as an artifact
        target_path = "target.csv"
        target.to_csv(target_path, index=False)
        mlflow.log_artifact(target_path, artifact_path="target")
        # Log basic info
        log_param("model", model)
        log_param("tune_hyperparams", tune_hyperparams)
        log_param("hyperparam_trials", hyperparam_trials)

        # split the data into train and test
        train_sample_size = int(0.9 * len(X))
        X_train, X_test = X[:train_sample_size], X[train_sample_size:]
        y_train, y_test = y[:train_sample_size], y[train_sample_size:]
        logger.info(f'Train sample size: {len(X_train)}')
        logger.info(f'Test sample size: {len(X_test)}')

        if not tune_hyperparams:
            logger.info('Using default hyperparameters')
            pipeline = make_pipeline(
                get_preprocessing_pipeline(),
                model_fn()
            )
        else:
            logger.info('Finding best hyperparameters with cross-validation')
            best_preprocessing_hyperparams, best_model_hyperparams, best_value = \
                find_best_hyperparams(model_fn, hyperparam_trials, X_train, y_train)
            mlflow.log_metric("cross_validation", best_value)
            logger.info(f'Best preprocessing hyperparameters: {best_preprocessing_hyperparams}')
            logger.info(f'Best model hyperparameters: {best_model_hyperparams}')

            log_param("best_preprocessing_hyperparams", best_preprocessing_hyperparams)
            log_param("best_model_hyperparams", best_model_hyperparams)

            pipeline = make_pipeline(
                get_preprocessing_pipeline(**best_preprocessing_hyperparams),
                model_fn(**best_model_hyperparams)
            )

        # train the model
        logger.info('Fitting model')
        pipeline.fit(X_train, y_train)

        # compute test MAE
        predictions = pipeline.predict(X_test)
        test_error = mean_absolute_error(y_test, predictions)
        logger.info(f'Test MAE: {test_error}')
        log_metric("Test MAE", test_error)

        # save the model to disk
        logger.info('Saving model to disk')
        model_path = MODELS_DIR / f'{model}_model.pkl'
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        # Log model as an artifact
        log_model(pipeline, "model")

        # Optional: Register the model if using MLflow Model Registry
        # mlflow.register_model("models:/<model-name>/<version>", "Production")

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='lasso')
    parser.add_argument('--tune-hyperparams', action='store_true')
    parser.add_argument('--sample-size', type=int, default=None)
    parser.add_argument('--hyperparam-trials', type=int, default=10)
    args = parser.parse_args()

    logger.info('Generating features and targets')
    features, target = transform_ts_data_into_features_and_target()

    if args.sample_size is not None:
        # reduce input size to speed up training
        features = features.head(args.sample_size)
        target = target.head(args.sample_size)
        
    logger.info('Training model')
    train(features, target,
          model=args.model,
          tune_hyperparams=args.tune_hyperparams,
          hyperparam_trials=args.hyperparam_trials
          )
