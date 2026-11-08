import joblib
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, Field

# Load the trained scikit-learn pipeline
MODEL_PATH = "mental_health_rf_pipeline.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(
        f"Model file '{MODEL_PATH}' not found. Ensure it exists in the working directory."
    )

app = FastAPI(
    title="Mental Health Trend Predictor API",
    description="REST API serving predictions from the Random Forest Regressor pipeline.",
    version="1.0.0",
)


# Define input schema matching exact dataframe feature column names
class MentalHealthMetrics(BaseModel):
    Code: str = Field(..., example="EGY")
    Schizophrenia_disorders: float = Field(
        ...,
        alias="Schizophrenia disorders (share of population)",
        example=0.2,
    )
    Depressive_disorders: float = Field(
        ..., alias="Depressive disorders (share of population)", example=3.5
    )
    Anxiety_disorders: float = Field(
        ..., alias="Anxiety disorders (share of population)", example=4.1
    )
    Bipolar_disorders: float = Field(
        ..., alias="Bipolar disorders (share of population)", example=0.7
    )

    class Config:
        populate_by_name = True


@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": True}


@app.post("/predict")
def predict_year(payload: MentalHealthMetrics):
    try:
        # Convert incoming JSON payload to Pandas DataFrame matching model features
        input_data = pd.DataFrame([payload.dict(by_alias=True)])

        # Run model pipeline inference
        prediction = model.predict(input_data)[0]

        return {
            "predicted_year": round(float(prediction), 2),
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Inference error: {str(e)}"
        )