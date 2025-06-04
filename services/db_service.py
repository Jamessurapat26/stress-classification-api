import os
import time
from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get MongoDB connection string from environment variables
MONGO_DETAILS = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_DETAILS)
database = client[os.getenv("DATABASE", "emotibit_data")]
sensor_collection = database[os.getenv("SENSOR_COLLECTION", "sensor_readings")]
preprocessed_collection = database[os.getenv("PREPROCESSED_COLLECTION", "preprocessed_data")]

def fix_objectid(doc):
    if doc and '_id' in doc:
        doc['_id'] = str(doc['_id'])
    return doc

class DBService:
    @staticmethod
    async def fetch_sensor_data(device_id: str, minutes: int = 6, projection=None) -> List[Dict[str, Any]]:
        time_threshold = int(time.time()) - (minutes * 60)
        if projection is None:
            projection = {
                "device_id": 1,
                "timestamp": 1,
                "sensors.eda": 1,
                "sensors.ppg": 1
            }
        query = {
            "device_id": device_id,
            "timestamp": {"$gte": time_threshold}
        }
        cursor = sensor_collection.find(query, projection).sort("timestamp", 1)
        cursor.batch_size(100)
        results = await cursor.to_list(length=None)
        return [fix_objectid(doc) for doc in results]

    @staticmethod
    async def save_preprocessed_data(data: dict) -> dict:
        result = await preprocessed_collection.insert_one(data)
        return {"inserted_id": str(result.inserted_id)}
    
    @staticmethod
    async def fetch_preprocessed_data(device_id: str, projection=None) -> List[Dict[str, Any]]:
        if projection is None:
            projection = {
                "deviceId": 1,
                "eda_features": 1,
                "ppg_features": 1,
                "hrv_indices": 1,
                "timestamp": 1
            }
        query = {"deviceId": device_id}
        # เพิ่ม .limit(5) ที่นี่เพื่อดึงข้อมูล 5 รายการล่าสุด
        cursor = preprocessed_collection.find(query, projection).sort("timestamp", -1).limit(5)
        cursor.batch_size(100)
        results = await cursor.to_list(length=None)
        return [fix_objectid(doc) for doc in results]

    @staticmethod
    async def get_HR(device_id: str) -> float:
        """
        Fetches the latest PPG_Rate value from preprocessed_data for a given deviceId.
        Returns only the numeric PPG_Rate value or None if not found.
        """
        projection = {
            "ppg_features.PPG_Rate": 1
        }
        query = {"deviceId": device_id}
        document = await preprocessed_collection.find_one(
            query, projection, sort=[("timestamp", -1)]
        )
        
        if document and "ppg_features" in document and "PPG_Rate" in document["ppg_features"]:
            return document["ppg_features"]["PPG_Rate"]
        return None