from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/xgb_churn.pkl")
scaler = joblib.load("models/scaler.pkl")

feature_columns = [col for col in scaler.feature_names_in_]

@app.post("/predict")
def predict(payload: dict):
    # Converting to DataFrame
    data = pd.DataFrame([payload])

    # Reindexing columns to expected format
    data = data.reindex(columns=feature_columns)

    # Applying scaling
    data_scaled = scaler.transform(data)

    # Prediction
    prob = model.predict_proba(data_scaled)[0][1]

    return {"probability_of_churn": float(prob)}
