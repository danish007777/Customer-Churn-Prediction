from fastapi import FastAPI
from app.schema import CustomerData
from app.predictor import pipeline
import pandas as pd

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer will churn.",
    version="1.0"
)

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is Running!"
    }

@app.post("/predict")
def predict(data: CustomerData):

    df = pd.DataFrame([data.dict()])

    prediction = pipeline.predict(df)

    result = "Yes" if prediction[0] == 1 else "No"

    return {
        "Prediction": result
    }