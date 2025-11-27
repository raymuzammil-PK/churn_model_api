from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Load model + scaler + final training columns list
model = joblib.load("models/xgb_churn.pkl")
scaler = joblib.load("models/scaler.pkl")

# This must match EXACT list you provided earlier
final_cols = [
 'SeniorCitizen','tenure','MonthlyCharges','TotalCharges','charge_ratio',
 'gender_Male','Partner_Yes','Dependents_Yes','PhoneService_Yes',
 'MultipleLines_No phone service','MultipleLines_Yes',
 'InternetService_Fiber optic','InternetService_No',
 'OnlineSecurity_No internet service','OnlineSecurity_Yes',
 'OnlineBackup_No internet service','OnlineBackup_Yes',
 'DeviceProtection_No internet service','DeviceProtection_Yes',
 'TechSupport_No internet service','TechSupport_Yes',
 'StreamingTV_No internet service','StreamingTV_Yes',
 'StreamingMovies_No internet service','StreamingMovies_Yes',
 'Contract_One year','Contract_Two year','PaperlessBilling_Yes',
 'PaymentMethod_Credit card (automatic)','PaymentMethod_Electronic check',
 'PaymentMethod_Mailed check','tenure_group_12-24','tenure_group_24-48',
 'tenure_group_48-72'
]

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str = None
    MultipleLines: str = None
    InternetService: str = None
    OnlineSecurity: str = None
    OnlineBackup: str = None
    DeviceProtection: str = None
    TechSupport: str = None
    StreamingTV: str = None
    StreamingMovies: str = None
    Contract: str = None
    PaperlessBilling: str = None
    PaymentMethod: str = None
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def home():
    return {"message": "Churn prediction API is running"}

@app.post("/predict")
def predict(customer: Customer):
    data = customer.dict()
    df = pd.DataFrame([data])

    # ===== FEATURE ENGINEERING =====

    # 1. Create charge_ratio automatically
    df["charge_ratio"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # 2. Create tenure_group exactly like training
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12","12-24","24-48","48-72"]
    )

    # 3. One-hot encode categorical features
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # ===== ALIGN WITH TRAINING COLUMNS =====

    # Add missing dummy columns (set to zero)
    for col in final_cols:
        if col not in df.columns:
            df[col] = 0

    # Keep only training columns & order them correctly
    df = df[final_cols]

    # ===== SCALE NUMERIC FEATURES =====

    numeric_cols = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges","charge_ratio"]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # ===== PREDICT =====
    prob = model.predict_proba(df)[0][1]

    return {"churn_probability": float(prob)}
