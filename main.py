from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib

app = FastAPI(
    title= "Deploy breast cancer model",
    version= "0.1.0"
)

@app.get("/")
async def home():
    return {"message": "Welcome to the API"}

@app.get("/api/v1/breast-cancer", tags=["breast-cancer"])
async def breast_cancer():
    return {"message": "breast cancer"}

@app.get("/api/v1/predict-breast-cancer", tags=["breast-cancer"])
async def predict_breast_cancer():
    return {"message": "Predict breast cancer"}

@app.get("/api/v1/predict-breast-cancer", tags=["breast-cancer"])
async def predict_breast_cancer():
    return {"message": "Predict breast cancer"}

@app.post("/api/v1/predict", tags=["predictions"])
async def predict():
    dictionary = {"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    try:
        df = pd.DataFrame(dictionary)
        prediction= model.predict(df)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"prediction": prediction.tolist()}
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
# Load the model

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi import status
import pandas as pd
import joblib

model = joblib.load("logistic_regression_model_v01.pkl")

@app.post("/api/v1/predict-breast-cancer", tags=["breast-cancer"])
async def predict(data: dict):
    dictionary = {"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    try:
        df = pd.DataFrame(dictionary)
        prediction= model.predict(df)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"prediction": prediction.tolist()}
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
