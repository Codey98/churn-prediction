from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load('model.pkl')


class CustomerFeatures(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


@app.post("/predict")
def predict(customer: CustomerFeatures):
    input_df = pd.DataFrame([customer.dict()])
    proba = model.predict_proba(input_df)[0, 1]
    prediction = "Yes" if proba >= 0.5 else "No"
    return {"churn_prediction": prediction, "churn_probability": round(float(proba), 4)}