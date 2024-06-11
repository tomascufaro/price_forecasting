import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
import sys

def load_model(run_id: str) -> Pipeline:
    """ Load the model from MLflow run """
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

def get_best_run(experiment_id: str) -> str:
    """ Get the run with the lowest Test MAE """
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=experiment_id,
        order_by=["metrics.Test MAE ASC"],
        max_results=1
    )
    best_run = runs[0]
    return best_run.info.run_id

def predict(model: Pipeline, X: pd.DataFrame) -> pd.Series:
    """ Make predictions using the loaded model """
    predictions = model.predict(X)
    return predictions

if __name__ == '__main__':
    # Set your MLflow tracking URI and experiment ID
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment_id = "1"  # Adjust your experiment ID

    # Get the best run from the experiment
    best_run_id = get_best_run(experiment_id)
    print(f"Best run ID: {best_run_id}")

    # Load the model from the best run
    model = load_model(best_run_id)
    print("Model loaded successfully.")

    # Example: Load new data for prediction
    # In practice, replace 'your_data.csv' with the path to your input data
    input_data_path = "your_data.csv"
    try:
        input_data = pd.read_csv(input_data_path)
        predictions = predict(model, input_data)
        print("Predictions:")
        print(predictions)
    except FileNotFoundError:
        print(f"Error: The file '{input_data_path}' does not exist.", file=sys.stderr)
