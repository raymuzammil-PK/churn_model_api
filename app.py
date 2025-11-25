from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running!"}

model = joblib.load("models/xgb_churn.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])

    df_scaled = scaler.transform(df)
    prob = model.predict_proba(df_scaled)[0][1]

    return {"probability": float(prob)}
