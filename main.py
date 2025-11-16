from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from pydantic import BaseModel
import logging
import os
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Barcelona Match Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load CatBoost model
try:
    model = joblib.load('cat_boost_new.pkl')   # Make sure this is the correct pkl
    logger.info("✅ Model loaded successfully from .pkl")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    model = None


# Response model
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
        csv_data = load_csv_data()
        if csv_data is None or csv_data.empty:
            raise HTTPException(status_code=404, detail="CSV data not available")

        # Prepare features (now does nothing extra)
        model_input = prepare_features(csv_data)

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
    PREVIOUSLY added last_updated column — now removed.
    The function simply returns the DataFrame as-is.
    """
    logger.info(f"✅ Feature shape before prediction: {input_df.shape}")
    return input_df


def make_prediction(model_input):
    """Make prediction using the loaded .pkl model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        predictions = model.predict(model_input)

        # Convert numpy output for JSON
        prediction_list = predictions[0].tolist() if hasattr(predictions[0], 'tolist') else predictions[0]

        return PredictionResponse(
            prediction=prediction_list,
            status="success",
            message=f"Predicted {len(prediction_list)} target values from CSV data"
        )

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model/features")
async def get_model_features():
    """Check what features the model expects"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Pipeline does NOT expose feature_names_in_ because ColumnTransformer is inside it
        # This endpoint may need updating later with preprocessing extraction
        if hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_.tolist()
        else:
            features = "Unable to extract feature names from wrapper"

        return {
            "expected_features": features,
            "expected_count": len(features) if isinstance(features, list) else 0
        }

    except Exception as e:
        return {"error": str(e)}


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
    uvicorn.run(app, host="0.0.0.0", port=8000)
