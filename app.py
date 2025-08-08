from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost as xgb
import pywt
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sonic Wave ML Model API",
    description="Deploys an XGBoost model to predict DTC and DTS values from well log data."
)

# Load pre-trained models
# Note: The models directory and files must exist for this to work
MODEL_DIR = "models"
DTC_MODEL_PATH = os.path.join(MODEL_DIR, 'dtc_model.joblib')
DTS_MODEL_PATH = os.path.join(MODEL_DIR, 'dts_model.joblib')

dtc_model = None
dts_model = None

@app.on_event("startup")
def load_models():
    """
    Loads the machine learning models on application startup.
    """
    global dtc_model, dts_model
    try:
        logger.info(f"Loading DTC model from {DTC_MODEL_PATH}")
        dtc_model = joblib.load(DTC_MODEL_PATH)
        logger.info("DTC model loaded successfully.")

        logger.info(f"Loading DTS model from {DTS_MODEL_PATH}")
        dts_model = joblib.load(DTS_MODEL_PATH)
        logger.info("DTS model loaded successfully.")

    except FileNotFoundError as e:
        logger.error(f"Failed to load a model file: {e}. Please ensure you have run 'python train_model.py' to generate the models.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during model loading: {e}")

# Serve the static HTML frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Define the input data model for validation
class WellLogData(BaseModel):
    CAL: float
    CNC: float
    GR: float
    HRD: float
    HRM: float
    PE: float
    ZDEN: float

# Preprocessing functions
def preprocess_input(data: WellLogData):
    """
    Applies the same preprocessing steps as the training script to a single
    input data point.
    """
    df = pd.DataFrame([data.dict()])
    
    # Replace negative placeholders with NaN and then impute
    df = df.replace(['-999', -999], np.nan)
    
    # Apply specific value cleaning rules from the notebook
    df.loc[df['GR'] < 0, 'GR'] = np.nan
    df.loc[df['CNC'] < 0, 'CNC'] = np.nan
    df.loc[df['PE'] < 0, 'PE'] = np.nan
    df.loc[df['ZDEN'] < 0, 'ZDEN'] = np.nan
    df.loc[df['GR'] > 250, 'GR'] = np.nan
    df.loc[df['CNC'] > 0.7, 'CNC'] = np.nan
    df.loc[df['HRD'] > 200, 'HRD'] = np.nan
    df.loc[df['HRM'] > 200, 'HRM'] = np.nan
    
    # Log transform HRD and HRM
    df['HRD'] = np.log(df['HRD'])
    df['HRM'] = np.log(df['HRM'])
    
    # Fill any remaining NaNs. These are default values based on the mean
    # of the training data after initial cleaning.
    df['CAL'] = df['CAL'].fillna(10.2)
    df['GR'] = df['GR'].fillna(68.0)
    df['CNC'] = df['CNC'].fillna(0.2)
    df['PE'] = df['PE'].fillna(6.8)
    df['ZDEN'] = df['ZDEN'].fillna(2.4)
    df['HRD'] = df['HRD'].fillna(1.8)
    df['HRM'] = df['HRM'].fillna(1.8)
    
    return df

def apply_wavelet_transformation_to_input(df):
    """
    Applies a wavelet transformation on the CNC feature.
    """
    # Create new columns for wavelet coefficients
    df['CNC_cD_level_4'] = 0.0
    df['CNC_cD_level_3'] = 0.0
    df['CNC_cD_level_2'] = 0.0
    df['CNC_cD_level_1'] = 0.0
    df['CNC_cA_level_4'] = 0.0
    
    # Apply wavelet transformation to the 'CNC' value
    coeffs = pywt.wavedec(df['CNC'].values, 'db4', level=4)
    
    df.loc[0, 'CNC_cA_level_4'] = coeffs[0][0]
    df.loc[0, 'CNC_cD_level_4'] = coeffs[1][0]
    df.loc[0, 'CNC_cD_level_3'] = coeffs[2][0]
    df.loc[0, 'CNC_cD_level_2'] = coeffs[3][0]
    df.loc[0, 'CNC_cD_level_1'] = coeffs[4][0]

    return df

@app.post("/predict")
def predict_sonic_wave(data: WellLogData):
    """
    Prediction endpoint that takes well log data and returns predicted DTC and DTS values.
    
    - **CAL**: Caliper log
    - **CNC**: Compensated Neutron Log
    - **GR**: Gamma Ray Log
    - **HRD**: Deep resistivity log
    - **HRM**: Medium resistivity log
    - **PE**: Photoelectric Effect Log
    - **ZDEN**: Density log
    
    The API returns a JSON object with the predicted 'DTC' and 'DTS' values.
    """
    
    if dtc_model is None or dts_model is None:
        return {"error": "Models are not loaded. Please ensure you have run 'python train_model.py' to generate the models."}
    
    # Preprocess the input data
    input_df = preprocess_input(data)
    
    # Apply wavelet transformation
    input_df = apply_wavelet_transformation_to_input(input_df)
    
    # Define features based on the trained model
    features = ['CAL', 'CNC', 'GR', 'HRD', 'HRM', 'PE', 'ZDEN', 
                'CNC_cD_level_4', 'CNC_cD_level_3', 'CNC_cD_level_2', 'CNC_cD_level_1', 
                'CNC_cA_level_4']
    
    # Make predictions and convert the numpy.float32 output to standard Python floats
    try:
        dtc_prediction = float(dtc_model.predict(input_df[features])[0])
        dts_prediction = float(dts_model.predict(input_df[features])[0])
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"error": f"Prediction failed due to an error: {e}"}
        
    return {"DTC": dtc_prediction, "DTS": dts_prediction}
