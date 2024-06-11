from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from mlflow.tracking import MlflowClient
import mlflow.pyfunc

app = FastAPI()

class PredictionRequest(BaseModel):
    # Define the structure of your incoming prediction data using Pydantic
    feature1: float
    feature2: float
    feature3: float

class PredictionResponse(BaseModel):
    predictions: list

# Load the model from MLflow at startup
def load_model_from_mlflow():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment_id = "1"  # Adjust your experiment ID
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=experiment_id,
        order_by=["metrics.Test MAE ASC"],
        max_results=1
    )
    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

model = load_model_from_mlflow()

@app.post("/predict", response_model=PredictionResponse)
def make_prediction(request: PredictionRequest):
    try:
        # Convert the request into a DataFrame
        input_data = pd.DataFrame([request.dict()])
        predictions = model.predict(input_data)
        return PredictionResponse(predictions=predictions.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
