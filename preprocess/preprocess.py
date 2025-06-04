from services.db_service import DBService
import pickle
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional, Any

class Preprocessor:
    """Preprocesses sensor data for LSTM model input."""
    
    DEFAULT_FEATURE_KEYS = ['EDA_Phasic', 'SCR_Amplitude', 'EDA_Tonic', 'SCR_Onsets']
    ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
    
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.scaler = None
        self.encoder = None
        self.feature_keys_order = self.DEFAULT_FEATURE_KEYS.copy()
        
        self._load_scaler()
        self._load_encoder()

    def _load_scaler(self) -> None:
        """Load the scaler from pickle file."""
        if self.scaler is not None:
            return
            
        scaler_path = self.ASSETS_DIR / "scaler.pkl"
        try:
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            logging.info("Scaler loaded successfully")
        except FileNotFoundError:
            logging.error(f"Scaler file not found at {scaler_path}")
        except Exception as e:
            logging.error(f"Failed to load scaler: {e}")
    
    def _load_encoder(self) -> None:
        """Load the encoder from JSON file."""
        if self.encoder is not None:
            return
            
        try:
            json_files = list(self.ASSETS_DIR.glob("*.json"))
            if not json_files:
                raise FileNotFoundError("No JSON encoder file found in assets directory")
                
            encoder_path = json_files[0]
            with open(encoder_path, "r") as f:
                self.encoder = json.load(f)
            logging.info("Encoder loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load encoder: {e}")

    async def fetch_recent_data(self, minutes: int = 5) -> List[Dict]:
        """Fetch recent sensor data."""
        return await DBService.fetch_sensor_data(self.device_id, minutes)

    async def save_preprocessed_data(self, data: Dict) -> Any:
        """Save preprocessed data to database."""
        return await DBService.save_preprocessed_data(data)

    def _get_recent_timesteps(self, data: List[Dict], num_timesteps: int) -> List[Dict]:
        """Get the most recent timesteps, sorted chronologically."""
        if len(data) < num_timesteps:
            raise ValueError(
                f"Not enough data for device {self.device_id}. "
                f"Need {num_timesteps} entries, got {len(data)}."
            )
        
        # Sort by timestamp (newest first) and take required number
        sorted_data = sorted(data, key=lambda x: x.get('timestamp', 0), reverse=True)
        recent_data = sorted_data[:num_timesteps]
        
        # Return in chronological order (oldest first) for LSTM
        return sorted(recent_data, key=lambda x: x.get('timestamp', 0))

    def _determine_feature_keys(self, sample_data: Dict) -> List[str]:
        """Determine feature keys from sample data if not predefined."""
        if self.feature_keys_order:
            return self.feature_keys_order
            
        feature_keys = []
        
        # Extract EDA features
        if sample_data.get('eda_features'):
            feature_keys.extend(sorted(sample_data['eda_features'].keys()))
            
        # Extract PPG features
        if sample_data.get('ppg_features'):
            feature_keys.extend(sorted(sample_data['ppg_features'].keys()))
            
        # Extract HRV features
        hrv_indices = sample_data.get('hrv_indices')
        if hrv_indices:
            if isinstance(hrv_indices, dict):
                feature_keys.extend(sorted(hrv_indices.keys()))
            elif isinstance(hrv_indices, pd.DataFrame) and not hrv_indices.empty:
                feature_keys.extend(sorted(hrv_indices.iloc[0].to_dict().keys()))
        
        if not feature_keys:
            raise ValueError(
                "Could not determine feature structure. "
                "Ensure eda_features, ppg_features, or hrv_indices exist."
            )
            
        return feature_keys

    def _extract_feature_value(self, doc: Dict, key: str) -> float:
        """Extract a single feature value from document."""
        # Check EDA features
        if doc.get('eda_features') and key in doc['eda_features']:
            return doc['eda_features'][key]
            
        # Check PPG features
        if doc.get('ppg_features') and key in doc['ppg_features']:
            return doc['ppg_features'][key]
            
        # Check HRV indices
        hrv_indices = doc.get('hrv_indices')
        if hrv_indices:
            if isinstance(hrv_indices, dict) and key in hrv_indices:
                return hrv_indices[key]
            elif (isinstance(hrv_indices, pd.DataFrame) and 
                  not hrv_indices.empty and 
                  key in hrv_indices.columns):
                return hrv_indices.iloc[0][key]
        
        return None

    def _extract_features_from_data(self, data: List[Dict]) -> pd.DataFrame:
        """Extract features from preprocessed data and return as DataFrame."""
        if not data:
            raise ValueError("No data provided for feature extraction")
            
        # Determine feature keys from first document
        self.feature_keys_order = self._determine_feature_keys(data[0])
        
        # Extract features for each timestep
        time_series_rows = []
        for doc in data:
            features = []
            for key in self.feature_keys_order:
                val = self._extract_feature_value(doc, key)
                
                # Handle None/NaN values
                if val is None or (isinstance(val, (float, np.float64)) and pd.isna(val)):
                    features.append(0.0)
                else:
                    features.append(float(val))
                    
            time_series_rows.append(features)
        
        return pd.DataFrame(time_series_rows, columns=self.feature_keys_order)

    def _add_personal_data(self, df: pd.DataFrame, personal_data: Any) -> pd.DataFrame:
        """Add personal data columns to DataFrame."""
        df = df.copy()
        
        # Handle both dictionary and object inputs
        if hasattr(personal_data, '__dict__'):
            # If it's an object, convert to dict or access attributes directly
            df['gender'] = getattr(personal_data, 'gender', '')
            df['bmi'] = getattr(personal_data, 'bmi', 0)
            df['sleep'] = getattr(personal_data, 'sleep', '')
            df['type'] = getattr(personal_data, 'skinType', '')
        elif isinstance(personal_data, dict):
            # If it's a dictionary, use .get() method
            df['gender'] = personal_data.get('gender', '')
            df['bmi'] = personal_data.get('bmi', 0)
            df['sleep'] = personal_data.get('sleep', '')
            df['type'] = personal_data.get('skinType', '')
        else:
            # Default values if personal_data is None or unexpected type
            df['gender'] = ''
            df['bmi'] = 0
            df['sleep'] = ''
            df['type'] = ''
            
        return df

    def _encode_categorical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical data using loaded encoder."""
        if not self.encoder:
            logging.warning("Encoder not loaded. Skipping categorical encoding.")
            return df
            
        df = df.copy()
        
        # Create reverse mappings
        categorical_mappings = {
            'gender': {v.lower(): int(k) for k, v in self.encoder.get('gender', {}).items()},
            'type': {v.lower(): int(k) for k, v in self.encoder.get('type', {}).items()},
            'sleep': {v.lower(): int(k) for k, v in self.encoder.get('sleep', {}).items()}
        }
        
        # Apply encodings
        for col, mapping in categorical_mappings.items():
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().map(mapping)
        
        # Fill missing mappings with 0
        return df.fillna(0)

    def _normalize_data(self, df: pd.DataFrame) -> np.ndarray:
        """Normalize data using loaded scaler."""
        if not self.scaler:
            logging.warning("Scaler not loaded. Skipping normalization.")
            return df.values
            
        return self.scaler.transform(df)

    def _reshape_for_lstm(self, data: np.ndarray, num_timesteps: int, num_features: int) -> np.ndarray:
        """Reshape data for LSTM input."""
        return data.reshape(1, num_timesteps, num_features)

    async def prepare_lstm_input(self, num_timesteps: int = 5, personal_data: Any = None) -> np.ndarray:
        """
        Prepare LSTM input from recent preprocessed data.
        
        Args:
            num_timesteps: Number of timesteps for LSTM sequence
            personal_data: Dictionary or object containing personal information
            
        Returns:
            NumPy array of shape (1, num_timesteps, num_features)
        """
        if personal_data is None:
            personal_data = {}
            
        # Fetch recent preprocessed data
        recent_data = await DBService.fetch_preprocessed_data(self.device_id)
        
        # Get required timesteps
        timestep_data = self._get_recent_timesteps(recent_data, num_timesteps)
        
        # Extract features into DataFrame
        features_df = self._extract_features_from_data(timestep_data)
        
        # Add personal data
        features_df = self._add_personal_data(features_df, personal_data)
        
        # Encode categorical data
        features_df = self._encode_categorical_data(features_df)
        
        # Normalize data
        normalized_data = self._normalize_data(features_df)
        
        # Calculate number of features (sensor + personal data)
        num_features = len(self.feature_keys_order) + 4  # 4 personal data fields
        
        # Reshape for LSTM
        return self._reshape_for_lstm(normalized_data, num_timesteps, num_features)