import pandas as pd
import numpy as np
import xgboost as xgb
import pywt
import joblib
import os

# Define a directory to save the trained models
MODEL_DIR = "models"

# Create the models directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Created directory: {MODEL_DIR}")

# --- Data Preprocessing Functions ---
def preprocess_data(df):
    """
    Cleans and prepares the raw well log data for model training.
    """
    # Replace placeholder values with NaN
    df = df.replace(['-999', -999], np.nan)

    # Clean specific features based on domain knowledge from the notebook
    df.loc[df['GR'] < 0, 'GR'] = np.nan
    df.loc[df['CNC'] < 0, 'CNC'] = np.nan
    df.loc[df['PE'] < 0, 'PE'] = np.nan
    df.loc[df['ZDEN'] < 0, 'ZDEN'] = np.nan
    df.loc[df['GR'] > 250, 'GR'] = np.nan
    df.loc[df['CNC'] > 0.7, 'CNC'] = np.nan
    df.loc[df['HRD'] > 200, 'HRD'] = np.nan
    df.loc[df['HRM'] > 200, 'HRM'] = np.nan

    # Apply log transformation to resistivity features
    df['HRD'] = np.log(df['HRD'])
    df['HRM'] = np.log(df['HRM'])

    # Impute missing values with the mean of each column
    df = df.fillna(df.mean())
    return df

def apply_wavelet_transformation(df):
    """
    Applies a wavelet transformation to the 'CNC' feature to extract more
    meaningful signals, creating new features for the model.
    """
    # Create new columns for the wavelet coefficients
    df['CNC_cD_level_4'] = 0.0
    df['CNC_cD_level_3'] = 0.0
    df['CNC_cD_level_2'] = 0.0
    df['CNC_cD_level_1'] = 0.0
    df['CNC_cA_level_4'] = 0.0
    
    # Iterate through each row and apply the wavelet transformation
    for i, row in df.iterrows():
        # Using 'db4' wavelet with 4 levels, as specified in the notebook
        coeffs = pywt.wavedec(np.array([row['CNC']]), 'db4', level=4)
        
        # Store the wavelet coefficients in the new columns
        df.loc[i, 'CNC_cA_level_4'] = coeffs[0][0]
        df.loc[i, 'CNC_cD_level_4'] = coeffs[1][0]
        df.loc[i, 'CNC_cD_level_3'] = coeffs[2][0]
        df.loc[i, 'CNC_cD_level_2'] = coeffs[3][0]
        df.loc[i, 'CNC_cD_level_1'] = coeffs[4][0]

    return df

# --- Main Training Script ---
if __name__ == "__main__":
    try:
        # Load the training data
        print("Loading training data...")
        train_df = pd.read_csv('train.csv')
        print("Training data loaded successfully.")
        
        # Apply data preprocessing and wavelet transformation
        print("Preprocessing data and applying wavelet transformation...")
        df = preprocess_data(train_df.copy())
        df = apply_wavelet_transformation(df)
        print("Data preparation complete.")

        # Define features and targets for the models
        features = ['CAL', 'CNC', 'GR', 'HRD', 'HRM', 'PE', 'ZDEN',
                    'CNC_cD_level_4', 'CNC_cD_level_3', 'CNC_cD_level_2', 'CNC_cD_level_1',
                    'CNC_cA_level_4']
        target_dtc = 'DTC'
        target_dts = 'DTS'

        X = df[features]
        y_dtc = df[target_dtc]
        y_dts = df[target_dts]

        # Initialize and train the DTC model using hyperparameters from the notebook
        print("Training DTC model...")
        xgb_model_dtc = xgb.XGBRegressor(random_state=42, max_depth=2, learning_rate=0.18, n_estimators=145, min_child_weight=6, gamma=0.3)
        xgb_model_dtc.fit(X, y_dtc)
        
        # Save the trained DTC model
        dtc_model_path = os.path.join(MODEL_DIR, 'dtc_model.joblib')
        joblib.dump(xgb_model_dtc, dtc_model_path)
        print(f"DTC model saved to {dtc_model_path}")

        # Initialize and train the DTS model using hyperparameters from the notebook
        print("Training DTS model...")
        xgb_model_dts = xgb.XGBRegressor(random_state=42, max_depth=7, learning_rate=0.19, n_estimators=135, min_child_weight=6, gamma=0.7)
        xgb_model_dts.fit(X, y_dts)
        
        # Save the trained DTS model
        dts_model_path = os.path.join(MODEL_DIR, 'dts_model.joblib')
        joblib.dump(xgb_model_dts, dts_model_path)
        print(f"DTS model saved to {dts_model_path}")
        
        print("\nAll models have been successfully trained and saved.")

    except FileNotFoundError as e:
        print(f"Error: Required file not found. Please ensure 'train.csv' is in the same directory.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during model training: {e}")
