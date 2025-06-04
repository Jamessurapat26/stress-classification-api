from pydantic import BaseModel, Field, validator
from typing import Optional, Literal, Dict, Any
from datetime import datetime

class InputData(BaseModel):
    """Input schema for stress prediction requests."""
    deviceId: str = Field(
        ..., 
        description="Device ID", 
        min_length=1, 
        max_length=50,
        pattern=r"^[a-zA-Z0-9_-]+$"  # Changed from regex to pattern
    )
    gender: Optional[Literal["male", "female"]] = Field(
        None, 
        description="Gender of the person"
    )
    bmi: Optional[float] = Field(
        None, 
        description="Body Mass Index", 
        ge=10.0, 
        le=50.0
    )
    sleep: Optional[Literal["6 - 8 hours", "Less than 8 hours", "More than 8 hours"]] = Field(
        None, 
        description="Sleep duration categories"
    )
    skinType: Optional[Literal["Type 2", "Type 3", "Type 4", "Type 5"]] = Field(
        None, 
        description="Skin type classification"
    )
    
    @validator('deviceId')
    def validate_device_id(cls, v):
        if not v.strip():
            raise ValueError('Device ID cannot be empty or whitespace')
        return v.strip()

class PredictionOutput(BaseModel):
    """Output schema for stress prediction responses."""
    device_id: str = Field(..., description="Device ID")
    predicted_class: int = Field(..., description="Predicted stress class (0-3)")
    predicted_label: str = Field(..., description="Human-readable stress level")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    confidence_percentage: float = Field(..., description="Confidence as percentage")
    all_predictions: Dict[str, float] = Field(..., description="All class predictions")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: int = Field(..., description="Unix timestamp")

class HeartRateResponse(BaseModel):
    """Response schema for heart rate requests."""
    device_id: str = Field(..., description="Device ID")
    heart_rate: Optional[float] = Field(None, description="Heart rate in BPM")
    timestamp: int = Field(..., description="Unix timestamp")
    status: str = Field(..., description="Response status")

class HealthCheckResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Model loading status")
    database_connected: bool = Field(..., description="Database connection status")
    timestamp: int = Field(..., description="Unix timestamp")