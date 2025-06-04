import os
import time
import logging
from typing import List, Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Custom exception for database operations."""
    pass

class DBService:
    """Enhanced database service with connection pooling and error handling."""
    
    _client: Optional[AsyncIOMotorClient] = None
    _database = None
    _sensor_collection = None
    _preprocessed_collection = None
    
    @classmethod
    async def initialize(cls):
        """Initialize database connection."""
        if cls._client is not None:
            return
            
        try:
            mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
            database_name = os.getenv("DATABASE", "emotibit_data")
            
            # Connection with optimized settings
            cls._client = AsyncIOMotorClient(
                mongo_uri,
                maxPoolSize=20,
                minPoolSize=5,
                maxIdleTimeMS=30000,
                waitQueueTimeoutMS=5000,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=20000,
            )
            
            cls._database = cls._client[database_name]
            cls._sensor_collection = cls._database[
                os.getenv("SENSOR_COLLECTION", "sensor_readings")
            ]
            cls._preprocessed_collection = cls._database[
                os.getenv("PREPROCESSED_COLLECTION", "preprocessed_data")
            ]
            
            # Test connection
            await cls._client.admin.command('ping')
            logger.info("Database connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")
    
    @classmethod
    async def close(cls):
        """Close database connection."""
        if cls._client:
            cls._client.close()
            cls._client = None
            logger.info("Database connection closed")
    
    @classmethod
    async def health_check(cls) -> bool:
        """Check database connectivity."""
        try:
            if cls._client is None:
                return False
            await cls._client.admin.command('ping')
            return True
        except Exception:
            return False
    
    @classmethod
    def _fix_objectid(cls, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ObjectId to string for JSON serialization."""
        if doc and '_id' in doc:
            doc['_id'] = str(doc['_id'])
        return doc
    
    @classmethod
    async def fetch_sensor_data(
        cls, 
        device_id: str, 
        minutes: int = 6, 
        projection: Optional[Dict[str, int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent sensor data for a device.
        
        Args:
            device_id: Device identifier
            minutes: Time window in minutes
            projection: MongoDB projection fields
            
        Returns:
            List of sensor data documents
        """
        try:
            if cls._sensor_collection is None:
                raise DatabaseError("Database not initialized")
            
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
            
            cursor = cls._sensor_collection.find(query, projection).sort("timestamp", 1)
            cursor.batch_size(100)
            
            # Add timeout for the query
            results = await asyncio.wait_for(
                cursor.to_list(length=None), 
                timeout=30.0
            )
            
            return [cls._fix_objectid(doc) for doc in results]
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching sensor data for device {device_id}")
            raise DatabaseError("Database query timeout")
        except PyMongoError as e:
            logger.error(f"MongoDB error fetching sensor data: {e}")
            raise DatabaseError(f"Database query failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching sensor data: {e}")
            raise DatabaseError(f"Unexpected database error: {e}")
    
    @classmethod
    async def fetch_preprocessed_data(
        cls, 
        device_id: str, 
        limit: int = 5,
        projection: Optional[Dict[str, int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent preprocessed data for a device.
        
        Args:
            device_id: Device identifier
            limit: Maximum number of records to return
            projection: MongoDB projection fields
            
        Returns:
            List of preprocessed data documents
        """
        try:
            if cls._preprocessed_collection is None:
                raise DatabaseError("Database not initialized")
            
            if projection is None:
                projection = {
                    "deviceId": 1,
                    "eda_features": 1,
                    "ppg_features": 1,
                    "hrv_indices": 1,
                    "timestamp": 1
                }
            
            query = {"deviceId": device_id}
            
            cursor = cls._preprocessed_collection.find(query, projection)\
                .sort("timestamp", -1)\
                .limit(limit)
            cursor.batch_size(100)
            
            results = await asyncio.wait_for(
                cursor.to_list(length=None), 
                timeout=30.0
            )
            
            return [cls._fix_objectid(doc) for doc in results]
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching preprocessed data for device {device_id}")
            raise DatabaseError("Database query timeout")
        except PyMongoError as e:
            logger.error(f"MongoDB error fetching preprocessed data: {e}")
            raise DatabaseError(f"Database query failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching preprocessed data: {e}")
            raise DatabaseError(f"Unexpected database error: {e}")
    
    @classmethod
    async def save_preprocessed_data(cls, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Save preprocessed data to database.
        
        Args:
            data: Preprocessed data to save
            
        Returns:
            Dictionary with inserted document ID
        """
        try:
            if cls._preprocessed_collection is None:
                raise DatabaseError("Database not initialized")
            
            # Add timestamp if not present
            if 'timestamp' not in data:
                data['timestamp'] = int(time.time())
            
            result = await asyncio.wait_for(
                cls._preprocessed_collection.insert_one(data),
                timeout=10.0
            )
            
            return {"inserted_id": str(result.inserted_id)}
            
        except asyncio.TimeoutError:
            logger.error("Timeout saving preprocessed data")
            raise DatabaseError("Database save timeout")
        except PyMongoError as e:
            logger.error(f"MongoDB error saving preprocessed data: {e}")
            raise DatabaseError(f"Database save failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving preprocessed data: {e}")
            raise DatabaseError(f"Unexpected database error: {e}")
    
    @classmethod
    async def get_HR(cls, device_id: str) -> Optional[float]:
        """
        Get the latest heart rate for a device.
        
        Args:
            device_id: Device identifier
            
        Returns:
            Latest heart rate value or None if not found
        """
        try:
            if cls._preprocessed_collection is None:
                raise DatabaseError("Database not initialized")
            
            projection = {"ppg_features.PPG_Rate": 1}
            query = {"deviceId": device_id}
            
            document = await asyncio.wait_for(
                cls._preprocessed_collection.find_one(
                    query, 
                    projection, 
                    sort=[("timestamp", -1)]
                ),
                timeout=10.0
            )
            
            if (document and "ppg_features" in document and 
                "PPG_Rate" in document["ppg_features"]):
                return document["ppg_features"]["PPG_Rate"]
            
            return None
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout getting heart rate for device {device_id}")
            raise DatabaseError("Database query timeout")
        except PyMongoError as e:
            logger.error(f"MongoDB error getting heart rate: {e}")
            raise DatabaseError(f"Database query failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error getting heart rate: {e}")
            raise DatabaseError(f"Unexpected database error: {e}")
