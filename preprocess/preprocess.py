from services.db_service import DBService
import pickle
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd


class Preprocessor:
    """Preprocesses sensor data for LSTM model input."""

    DEFAULT_FEATURE_KEYS = ['EDA_Phasic', 'SCR_Amplitude', 'EDA_Tonic', 'SCR_Onsets']
    ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

    _scaler = None
    _encoder = None

    def __init__(self, device_id: str):
        self.device_id = device_id
        self.feature_keys_order = self.DEFAULT_FEATURE_KEYS.copy()
        self._ensure_assets_loaded()

    @classmethod
    def _ensure_assets_loaded(cls):
        if cls._scaler is None:
            cls._scaler = cls._load_pickle(cls.ASSETS_DIR / "scaler.pkl", "Scaler")
        if cls._encoder is None:
            cls._encoder = cls._load_json_encoder()

    @classmethod
    def _load_pickle(cls, path: Path, name: str) -> Optional[Any]:
        try:
            with path.open("rb") as f:
                obj = pickle.load(f)
            logging.info(f"{name} loaded successfully from {path}")
            return obj
        except FileNotFoundError:
            logging.error(f"{name} file not found at {path}")
        except Exception as e:
            logging.exception(f"Failed to load {name}: {e}")
        return None

    @classmethod
    def _load_json_encoder(cls) -> Optional[Dict]:
        try:
            json_path = next(cls.ASSETS_DIR.glob("*.json"))
            with json_path.open("r") as f:
                encoder = json.load(f)
            logging.info(f"Encoder loaded successfully from {json_path}")
            return encoder
        except StopIteration:
            logging.error("No JSON encoder file found in assets directory")
        except Exception as e:
            logging.exception(f"Failed to load encoder: {e}")
        return None

    async def fetch_recent_data(self, minutes: int = 5) -> List[Dict]:
        return await DBService.fetch_sensor_data(self.device_id, minutes)

    async def save_preprocessed_data(self, data: Dict) -> Any:
        return await DBService.save_preprocessed_data(data)

    def _get_recent_timesteps(self, data: List[Dict], num_timesteps: int) -> List[Dict]:
        if len(data) < num_timesteps:
            raise ValueError(
                f"Not enough data for device {self.device_id}. "
                f"Need {num_timesteps}, got {len(data)}."
            )
        sorted_data = sorted(data, key=lambda x: x.get('timestamp', 0), reverse=True)
        return sorted(sorted_data[:num_timesteps], key=lambda x: x.get('timestamp', 0))

    def _determine_feature_keys(self, sample: Dict) -> List[str]:
        if self.feature_keys_order:
            return self.feature_keys_order

        keys = []
        for key in ['eda_features', 'ppg_features']:
            if sample.get(key):
                keys.extend(sorted(sample[key].keys()))

        hrv = sample.get('hrv_indices')
        if isinstance(hrv, dict):
            keys.extend(sorted(hrv.keys()))
        elif isinstance(hrv, pd.DataFrame) and not hrv.empty:
            keys.extend(sorted(hrv.iloc[0].to_dict().keys()))

        if not keys:
            raise ValueError("No valid features found in sample data.")

        logging.info(f"Determined feature keys: {keys}")
        return keys

    def _extract_feature_value(self, doc: Dict, key: str) -> float:
        for section in ['eda_features', 'ppg_features']:
            if key in doc.get(section, {}):
                return doc[section][key]

        hrv = doc.get('hrv_indices')
        if isinstance(hrv, dict):
            return hrv.get(key, 0.0)
        if isinstance(hrv, pd.DataFrame) and key in hrv.columns:
            return hrv.iloc[0].get(key, 0.0)
        return 0.0

    def _extract_features_from_data(self, data: List[Dict]) -> pd.DataFrame:
        if not data:
            raise ValueError("Empty input data.")
        self.feature_keys_order = self._determine_feature_keys(data[0])
        rows = [
            [float(self._extract_feature_value(doc, key) or 0.0) for key in self.feature_keys_order]
            for doc in data
        ]
        df = pd.DataFrame(rows, columns=self.feature_keys_order)
        logging.info(f"Extracted DataFrame shape: {df.shape}")
        return df

    def _add_personal_data(self, df: pd.DataFrame, p: Union[Dict, Any]) -> pd.DataFrame:
        if not isinstance(p, dict) and not hasattr(p, '__dict__'):
            p = {}

        def get(val, default=''):
            return p.get(val, default) if isinstance(p, dict) else getattr(p, val, default)

        df['gender'] = get('gender', '')
        df['bmi'] = float(get('bmi', 0))
        df['sleep'] = get('sleep', '')
        df['type'] = get('skinType', '')
        return df

    def _encode_categorical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._encoder:
            logging.warning("Encoder not loaded. Skipping encoding.")
            return df

        df = df.copy()
        for col in ['gender', 'type', 'sleep']:
            if col in df.columns:
                mapping = {v.lower(): int(k) for k, v in self._encoder.get(col, {}).items()}
                df[col] = df[col].astype(str).str.lower().map(mapping).fillna(0).astype(int)
        return df

    def _normalize_data(self, df: pd.DataFrame) -> np.ndarray:
        if not self._scaler:
            logging.warning("Scaler not loaded. Skipping normalization.")
            return df.values
        return self._scaler.transform(df)

    def _reshape_for_lstm(self, data: np.ndarray, timesteps: int, features: int) -> np.ndarray:
        return data.reshape(1, timesteps, features)

    async def prepare_lstm_input(self, num_timesteps: int = 5, personal_data: Optional[Union[Dict, Any]] = None) -> np.ndarray:
        personal_data = personal_data or {}
        data = await DBService.fetch_preprocessed_data(self.device_id)
        timesteps_data = self._get_recent_timesteps(data, num_timesteps)
        df = self._extract_features_from_data(timesteps_data)
        df = self._add_personal_data(df, personal_data)
        df = self._encode_categorical_data(df)
        normalized = self._normalize_data(df)
        num_features = len(self.feature_keys_order) + 4  # gender, bmi, sleep, type
        reshaped = self._reshape_for_lstm(normalized, num_timesteps, num_features)
        logging.info(f"LSTM input shape: {reshaped.shape}")
        return reshaped
