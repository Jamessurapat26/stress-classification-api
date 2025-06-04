# predictEndpoint.py
import logging
import time
from fastapi import APIRouter, HTTPException
from schema.predictSchema import PredictionOutput, InputData
from preprocess.preprocess import Preprocessor
from services.db_service import DBService
from model.model import Model
import pandas as pd
import neurokit2 as nk

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/getHrByDevice_id/{deviceId}", response_model=float)
async def get_hr_by_device_id(deviceId: str):
    """
    Fetch the latest HR value for a given device ID.
    """
    return await DBService.get_HR(deviceId)


@router.post("/predict-lstm", response_model=dict)
async def predict_stress_lstm(data: InputData):
    """
    Predict stress level from LSTM model using recent sensor data and personal info.
    """
    logger.info(f"Received prediction request for device {data.deviceId}")
    processor = Preprocessor(data.deviceId)

    try:
        model_input = await processor.prepare_lstm_input(num_timesteps=5, personal_data=data)
    except ValueError as ve:
        logger.warning(f"Data error for device {data.deviceId}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"Unexpected error for device {data.deviceId}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    logger.info(f"Model input shape: {model_input.shape}")
    
    model = Model()
    result = model.predict(model_input)

    best_class = max(result, key=result.get)
    best_confidence = result[best_class]

    class_labels = {
        "0": "High",
        "1": "Low",
        "2": "Medium",
        "3": "Normal"
    }
    predicted_label = class_labels.get(best_class, f"Class {best_class}")

    logger.info(
        f"Prediction for {data.deviceId}: Class={best_class} "
        f"({predicted_label}), Confidence={best_confidence:.2%}"
    )

    return {
        "deviceId": data.deviceId,
        "predicted_class": int(best_class),
        "predicted_label": predicted_label,
        "confidence": round(best_confidence, 4),
        "confidence_percentage": round(best_confidence * 100, 2),
        "all_predictions": {k: round(v, 4) for k, v in result.items()},
        "timestamp": int(time.time())
    }


@router.get("/processHRByDevice_id/{deviceId}", response_model=dict)
async def process_hr_by_device_id(deviceId: str):
    """
    Process raw PPG/EDA signals, extract features, and save to the database.
    """
    processor = Preprocessor(deviceId)
    data = await processor.fetch_recent_data()

    merged = {
        "deviceId": deviceId,
        "eda": [],
        "ppg": [],
        "timestamps": []
    }

    for doc in data:
        sensors = doc.get("sensors", {})
        merged["eda"].extend(sensors.get("eda", []))
        merged["ppg"].extend(sensors.get("ppg", []))
        if "timestamp" in doc:
            merged["timestamps"].append(doc["timestamp"])

    if not merged["ppg"] or not merged["eda"]:
        raise HTTPException(status_code=400, detail="Not enough PPG or EDA data")

    try:
        ppg_signals, _ = nk.ppg_process(merged["ppg"], sampling_rate=100, heart_rate=True)
        eda_signals, _ = nk.eda_process(merged["eda"], sampling_rate=15)
        hrv_indices = nk.hrv(ppg_signals["PPG_Peaks"], sampling_rate=100)
    except Exception as e:
        logger.exception(f"Signal processing failed for device {deviceId}: {e}")
        raise HTTPException(status_code=500, detail="Signal processing failed")

    resampled_ppg = ppg_signals.tail(100).mean()
    resampled_eda = eda_signals.tail(15).mean()
    hrv_dict = hrv_indices.iloc[0].to_dict() if not hrv_indices.empty else {}

    processed_data = {
        "deviceId": deviceId,
        "eda_features": {k: None if pd.isna(v) else v for k, v in resampled_eda.items()},
        "ppg_features": {k: None if pd.isna(v) else v for k, v in resampled_ppg.items()},
        "hrv_indices": {k: None if pd.isna(v) else v for k, v in hrv_dict.items()},
        "timestamp": int(time.time())
    }

    db_result = await processor.save_preprocessed_data(processed_data)
    success = "inserted_id" in db_result

    if success:
        logger.info(f"Saved processed data for {deviceId} with ID: {db_result['inserted_id']}")
        return {
            "status": "success",
            "device_id": deviceId,
            "HR": processed_data["ppg_features"].get("PPG_Rate", None)
        }
    else:
        logger.warning(f"Failed to save processed data for {deviceId}")
        return {"status": "error"}
