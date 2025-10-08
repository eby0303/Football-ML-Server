from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Any
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Football Analytics Model API",
    description="API for predicting football match statistics using CatBoost model",
    version="1.0.0"
)

# Load the model
try:
    model = joblib.load("best_cat_boost.pkl")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise e

# Define target columns (same as in your training)
TARGET_COLUMNS = [
    "For_GF", "For_GA", "For_Result", "Expected_xG", "Expected_npxG",
    "Standard_Gls", "Standard_Sh", "Standard_SoT", "Standard_G/Sh", "Standard_G/SoT",
    "Performance_PSxG", "Tackles_Tkl", "Tackles_TklW", "Challenges_Tkl", "Challenges_Tkl%",
    "Unnamed_24_Clr", "Unnamed_23_Tkl+Int", "Unnamed_31_PrgP", "Total_PrgDist", "Total_TotDist",
    "Carries_PrgDist", "Carries_Carries", "Touches_Touches", "Total_Cmp%", "Short_Cmp%",
    "Medium_Cmp%", "Long_Cmp%", "Passes_Launch%", "For_Poss"
]

# Define input schema based on your dataset features
class FootballMatchInput(BaseModel):
    Tackles_TklW: float
    Goal_Kicks_Att: float
    For_Day: str
    Pass_Types_Dead: float
    Tackles_Att_3rd: float
    Pass_Types_TB: float
    Passes_Launch: float
    Medium_Cmp: float
    Goal_Kicks_AvgLen: float
    Challenges_Tkl: float
    Tackles_Def_3rd: float
    Medium_Att: float
    Long_Att: float
    Standard_Dist: float
    Performance_PKwon: float
    Unnamed_23_Tkl_Int: float
    Penalty_Kicks_PKA: float
    For_Time: str
    Challenges_Tkl_percent: float
    Passes_AvgLen: float
    Performance_Fld: float
    Take_Ons_Tkld: float
    Take_Ons_Succ: float
    Passes_Att_GK: float
    Carries_CPA: float
    Performance_OG: float
    Challenges_Lost: float
    For_Venue: str
    For_Date: str
    For_Round: str
    last_updated: str
    Penalty_Kicks_PKatt: float
    Carries_1_3: float
    Medium_Cmp_percent: float
    Expected_G_xG: float
    Expected_np_G_xG: float
    Performance_CS: float
    Performance_Save_percent: float
    Performance_Saves: float
    Unnamed_24_Clr: float
    Unnamed_25_Err: float
    Unnamed_31_PrgP: float
    Carries_PrgDist: float
    Carries_Carries: float
    Touches_Touches: float
    Passes_Thr: float
    Short_Cmp_percent: float
    Long_Cmp_percent: float

# Response model
class PredictionResponse(BaseModel):
    predictions: Dict[str, float]
    input_features: Dict[str, Any]

@app.get("/")
async def root():
    return {"message": "Football Analytics Model API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: FootballMatchInput):
    """
    Make predictions using the trained CatBoost model
    """
    try:
        # Convert input to dictionary and prepare for DataFrame
        input_dict = input_data.dict()
        
        # Handle column name differences
        input_dict["Unnamed_23_Tkl+Int"] = input_dict.pop("Unnamed_23_Tkl_Int")
        input_dict["Challenges_Tkl%"] = input_dict.pop("Challenges_Tkl_percent")
        input_dict["Medium_Cmp%"] = input_dict.pop("Medium_Cmp_percent")
        input_dict["Short_Cmp%"] = input_dict.pop("Short_Cmp_percent")
        input_dict["Long_Cmp%"] = input_dict.pop("Long_Cmp_percent")
        input_dict["Carries_1/3"] = input_dict.pop("Carries_1_3")
        input_dict["Passes_Launch%"] = input_dict.pop("Passes_Launch")
        
        # Create DataFrame with proper column order
        features_df = pd.DataFrame([input_dict])
        
        # Make prediction
        predictions = model.predict(features_df)
        
        # Convert predictions to dictionary with target names
        predictions_dict = {
            target: float(pred) 
            for target, pred in zip(TARGET_COLUMNS, predictions[0])
        }
        
        logger.info("Prediction successful")
        
        return PredictionResponse(
            predictions=predictions_dict,
            input_features=input_data.dict()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# @app.get("/targets")
# async def get_target_columns():
#     """
#     Get the list of target columns the model predicts
#     """
#     return {"target_columns": TARGET_COLUMNS}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)