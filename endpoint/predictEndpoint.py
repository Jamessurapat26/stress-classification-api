# predictEndpoint.py
import logging
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder # import jsonable_encoder เข้ามา
from schema.predictSchema import PredictionOutput, InputData
from preprocess.preprocess import Preprocessor
import pandas as pd
import numpy as np
import neurokit2 as nk
import random
import time
import json
from pathlib import Path
from services.db_service import DBService
from model.model import Model

router = APIRouter()

@router.get("/getHrByDevice_id/{deviceId}", response_model=float)
async def get_hr_by_device_id(deviceId: str):
    """
    Fetches the latest 1 PPG_Rate data point for a given deviceId.
    """
    return await DBService.get_HR(deviceId)

# @router.post("/", response_model=dict)
# async def predict_stress(data: InputData):
#     input_df = pd.DataFrame([data.dict()])
#     prediction = model.predict(input_df.to_numpy())
#     predicted_class = int(np.argmax(prediction, axis=1)[0])
#     return {"predicted_stress": predicted_class}

@router.post("/predict-lstm", response_model=dict)
async def predict_stress_lstm(data: InputData):

    print(f"Received data for device {data.deviceId}: {data}")
    
    processor  = Preprocessor(data.deviceId)
    try:
        model_input = await processor.prepare_lstm_input(num_timesteps=5, personal_data=data)
    except ValueError as e:
        # จับข้อผิดพลาดที่ Preprocessor อาจจะโยนมา (เช่น ข้อมูลไม่พอ)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # จับข้อผิดพลาดอื่นๆ ที่อาจเกิดขึ้น
        logging.error(f"Error preparing LSTM input for device {data.deviceId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

    print(f"Model input shape: {model_input.shape}")
    model = Model()
    model.print_model_summary()  # แสดงสรุปของโมเดล
    result = model.predict(model_input)  # ทดสอบการทำงานของโมเดล

    print(f"Model prediction result: {result}")
    # Find the best prediction result
    best_class = max(result, key=result.get)
    best_confidence = result[best_class]
    
    print(f"Best prediction - Class: {best_class}, Confidence: {best_confidence:.4f} ({best_confidence*100:.2f}%)")
    
    # Optional: Define class labels for better readability
    class_labels = {
        "0": "High",
        "1": "Low",
        "2": "Medium",
        "3": "normal"
    }
    
    predicted_label = class_labels.get(best_class, f"Class {best_class}")
    print(f"Predicted stress level: {predicted_label} with {best_confidence*100:.2f}% confidence")

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
    processor = Preprocessor(deviceId)
    data = await processor.fetch_recent_data()
    merged = {"deviceId": deviceId, "eda": [], "ppg": [], "timestamps": []}

    for doc in data:
        sensors = doc.get('sensors', {})
        merged['eda'].extend(sensors.get('eda', []))
        merged['ppg'].extend(sensors.get('ppg', []))
        if 'timestamp' in doc:
            merged['timestamps'].append(doc['timestamp'])

    ppg, eda = merged["ppg"], merged["eda"]
    if not ppg or not eda:
        raise HTTPException(status_code=400, detail="Not enough PPG/EDA data")

    ppg_signals, _ = nk.ppg_process(ppg, sampling_rate=100, heart_rate=True)
    eda_signals, _ = nk.eda_process(eda, sampling_rate=15)
    ppg_peak = ppg_signals['PPG_Peaks']
    hrv_indices = nk.hrv(ppg_peak, sampling_rate=100)
    resampled_ppg = ppg_signals.tail(100).mean()
    resampled_eda = eda_signals.tail(15).mean()

    processed_data = {
        "deviceId": deviceId,
        "eda_features": {k: None if pd.isna(v) else v for k, v in resampled_eda.items()},
        "ppg_features": {k: None if pd.isna(v) else v for k, v in resampled_ppg.items()},
        "hrv_indices": {k: None if pd.isna(v) else v for k, v in (hrv_indices.iloc[0].to_dict() if not hrv_indices.empty else {}).items()},
        "timestamp": int(time.time())
    }
    db_result = await processor.save_preprocessed_data(processed_data)

    if "inserted_id" in db_result:
        processed_data["db_status"] = "success"
        processed_data["inserted_id"] = db_result["inserted_id"] # ใช้ค่าที่ได้จาก db_result โดยตรง
        print(f"Data saved to database with ID: {processed_data['inserted_id']}")
        print(processed_data) # แสดงข้อมูลที่ถูกบันทึก
        return {
            "status": "success",
            "device_id": deviceId,
            "HR": processed_data["ppg_features"].get("PPG_Rate", None),
        } 
    else:
        return {"status": "error"}
