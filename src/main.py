from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from src.schemas import PredictRequest, PredictResponse
from src.predict import make_prediction
from src.projection import get_projection
from src.metrics import get_metrics

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        # Extract features explicitly
        features = request.features
        result = make_prediction(features, request.model_name)
        return PredictResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/projection")
def projection(method: str = Query("pca", enum=["pca", "tsne"])):
    try:
        return get_projection(method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics/{model_name}")
def metrics(model_name: str):
    try:
        if model_name.lower() not in ["knn", "svm", "nb"]:
            raise ValueError("Invalid model name. Choose from: knn, svm, nb.")

        return get_metrics(model_name.lower())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
