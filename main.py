from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib

# Load the model
model = joblib.load("model/logistic_regression_model_v01.pkl")

app = FastAPI(
    title= "Deploy breast cancer model",
    version= "0.1.0"
)

class Item(BaseModel):
    data: list

@app.get("/")
async def home():
    return {"message": "Welcome to the API"}

@app.get("/api/v1/breast-cancer", tags=["breast-cancer"])
async def breast_cancer():
    return {"message": "breast cancer"}

@app.get("/api/v1/predict-breast-cancer", tags=["breast-cancer"])
async def predict_breast_cancer():
    return {"message": "Predict breast cancer"}

@app.post("/api/v1/predict", tags=["predictions"])
async def predict(item: Item):
    try:
        df = pd.DataFrame([item.data], columns=[f'feature_{i}' for i in range(len(item.data))])
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

@app.post("/api/v1/predict-breast-cancer", tags=["breast-cancer"])
async def predict_breast_cancer(item: Item):
    try:
        df = pd.DataFrame([item.data], columns=[f'feature_{i}' for i in range(len(item.data))])
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