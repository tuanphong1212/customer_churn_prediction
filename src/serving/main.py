from contextlib import asynccontextmanager

import pandas as pd
import mlflow
from fastapi import FastAPI , HTTPException
from feast import FeatureStore
from mlflow.models import get_model_info
from pydantic import BaseModel

MLFLOW_TRACKING_URL = "http://127.0.0.1:5000"
FEATURE_REPO_PATH = "feature_repo/feature_repo"
MODEL_URI = "models:/customer_churn_model@champion"
FEATURE_SERVICE_NAME = "customer_churn_v1"

state = {}

@asynccontextmanager

async def lifespan(app : FastAPI):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)

    model_info = get_model_info(MODEL_URI)
    state["feature_columns"] = model_info.signature.inputs.input_names()
    state["model"] = mlflow.sklearn.load_model(MODEL_URI)
    state["store"] = FeatureStore(repo_path=FEATURE_REPO_PATH)

    print(f"Model loaded. Expecting {len(state['feature_columns'])} feature columns.")
    yield
    state.clear()

app = FastAPI(title="Customer Churn Prediction API" , lifespan=lifespan)

class PredictRequest(BaseModel):
    customer_id : int

class PredictResponse(BaseModel):
    customer_id : int
    churn_prediction : int
    churn_probability : float

@app.get('/health')
async def health():
    return {"status" : "ok"}

@app.post("/predict" , response_model=PredictResponse)
async def Predict(request : PredictRequest):
    store : FeatureStore = state['store']
    feature_service = store.get_feature_service(FEATURE_SERVICE_NAME)

    features_df = store.get_online_features(
        features=feature_service,
        entity_rows=[{"CustomerID" : request.customer_id}]
    ).to_df()

    feature_columns = state["feature_columns"]
    missing_or_null = features_df[feature_columns].isnull().any(axis=1).iloc[0]

    if missing_or_null:
        raise HTTPException(
            status_code=404,
            detail=f"CustomerID {request.customer_id} không tồn tại hoặc thiếu feature trong online store."
        )

    X = features_df[feature_columns]

    model = state["model"]
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0 , 1])

    return PredictResponse(
        customer_id = request.customer_id,
        churn_prediction = prediction,
        churn_probability = round(probability , 4)
    )