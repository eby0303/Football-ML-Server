from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from pydantic import BaseModel
import logging
import os
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware


# Initialize FastAPI app
app = FastAPI(title="Barcelona Match Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",      # Your React frontend
        "http://127.0.0.1:8080"       # Alternative localhost
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load your existing CatBoost model from .pkl
try:
    model = joblib.load('cat_boost.pkl')  # Your existing .pkl file
    logger.info("✅ Model loaded successfully from .pkl")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    model = None

class PredictionResponse(BaseModel):
    prediction: list
    status: str
    message: str = ""

@app.get("/")
async def root():
    return {"message": "Barcelona Match Predictor API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/predict/latest", response_model=PredictionResponse)
async def predict_latest_match():
    """
    Load data from CSV and return prediction
    """
    try:
        # Load data from CSV
        csv_data = load_csv_data()
        if csv_data is None or csv_data.empty:
            raise HTTPException(status_code=404, detail="CSV data not available")
        
        # Prepare features for model
        model_input = prepare_features(csv_data)
        
        # Make prediction
        prediction_result = make_prediction(model_input)
        
        return prediction_result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def load_csv_data():
    """Load data from CSV file"""
    try:
        if os.path.exists("barca_final_25_26.csv"):
            df = pd.read_csv("barca_final_25_26.csv")
            logger.info(f"✅ Loaded CSV data with shape: {df.shape}")
            return df
        else:
            logger.error("❌ CSV file not found")
            return None
    except Exception as e:
        logger.error(f"❌ Failed to load CSV: {e}")
        return None

def prepare_features(input_df):
    """
    Add missing columns that the model expects
    """
    # Missing columns with default values
    missing_columns = {
        'Performance_CS': 0,           # Clean Sheets (default: 0)
        'Performance_Save%': 0.0,      # Save Percentage (default: 0.0)
        'Performance_Saves': 0,        # Saves (default: 0)
        'Passes_Thr': 0,               # Through Passes (default: 0)
        'Unnamed_25_Err': 0,           # Error column (default: 0)
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Current timestamp
    }
    
    # Add missing columns with default values
    for col, default_value in missing_columns.items():
        if col not in input_df.columns:
            input_df[col] = default_value
            logger.info(f"✅ Added missing column: {col} = {default_value}")
    
    logger.info(f"✅ Final feature shape: {input_df.shape}")
    return input_df

def make_prediction(model_input):
    """Make prediction using the loaded .pkl model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Your model pipeline should handle all preprocessing
        predictions = model.predict(model_input)
        
        # Convert numpy array to list for JSON serialization
        prediction_list = predictions[0].tolist() if hasattr(predictions[0], 'tolist') else predictions[0]
        
        return PredictionResponse(
            prediction=prediction_list,
            status="success",
            message=f"Predicted {len(prediction_list)} target values from CSV data"
        )
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Additional endpoint to check model features
@app.get("/model/features")
async def get_model_features():
    """Check what features the model expects"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Try to get feature names from the model
        if hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_.tolist()
        else:
            features = "Unable to extract feature names"
        
        return {
            "expected_features": features,
            "expected_count": len(features) if isinstance(features, list) else 0
        }
    except Exception as e:
        return {"error": str(e)}

# Additional endpoint for testing with custom data
@app.post("/predict/custom")
async def predict_custom(input_data: dict):
    """
    Make prediction with custom feature input
    """
    try:
        input_df = pd.DataFrame([input_data])
        prepared_input = prepare_features(input_df)
        prediction_result = make_prediction(prepared_input)
        return prediction_result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)