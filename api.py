from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os
import warnings
from typing import Optional


warnings.filterwarnings("ignore")
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
  
]


MODEL_PATH = 'models/accident_predictor_sarima.joblib'

models = {}

def load_models():
    global models
    if os.path.exists(MODEL_PATH):
        try:
            models = joblib.load(MODEL_PATH)
            print(f"Models loaded successfully from {MODEL_PATH}")
            print(f"Available categories: {list(models.keys())}")
        except Exception as e:
            print(f"Error loading models: {e}")
            models = {} 
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
        models = {}

app = FastAPI(
    title="API",
    description="Moaksh Kakkar's submission for DPS Challenge",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

class PredictionRequest(BaseModel):
    category: Optional[str] = Field(default="Alkoholunfälle")
    year: int = Field(..., gt=1999)
    month: int = Field(..., ge=1, le=12)

class PredictionResponse(BaseModel):
    category: str
    year: int
    month: int
    predicted_value: float

@app.on_event("startup")
async def startup_event():
    print("Starting API and loading models...")
    load_models()

@app.post("/predict", response_model=PredictionResponse)
async def predict_accidents(request: PredictionRequest, category = "Alkoholunfälle"):
    category = request.category
    year = request.year
    month = request.month

    
    try:
        predict_date_str = f"{year}-{month:02d}-01"
        predict_date = pd.to_datetime(predict_date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid year/month combination.")

    model = models[category]

    try:
        prediction = model.predict(start=predict_date, end=predict_date)
        predicted_value = prediction.iloc[0] 
        if predicted_value < 0:
            predicted_value = 0.0

    except Exception as e:
        print(f"Error during prediction for {category} at {predict_date_str}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed for {category}. Error: {e}")

    return PredictionResponse(
        category=category,
        year=year,
        month=month,
        predicted_value=predicted_value
    )
@app.get("/")
async def read_root():
    return {"message": "Moaksh Kakkar's submission for DPS Challenge. Go to /docs for API documentation"}
