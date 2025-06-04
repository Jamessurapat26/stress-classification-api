from pydantic import BaseModel, Field
from typing import Optional, Literal

class PredictionOutput(BaseModel):
    deviceId: str = Field(..., description="Device ID")
    stressPrediction: Literal["normal", "low", "medium", "high"]
    timeStamp: int

class InputData(BaseModel):
    deviceId: str = Field(..., description="Device ID", min_length=1, max_length=50)
    gender: Optional[Literal["male", "female"]] = Field(None, description="Gender of the person")
    bmi: Optional[float] = Field(None, description="Body Mass Index", ge=10.0, le=50.0)
    sleep: Optional[Literal["6 - 8 hours", "Less than 8 hours", "More than 8 hours"]] = Field(None, description="Sleep duration categories")
    skinType: Optional[Literal["Type 2", "Type 3", "Type 4", "Type 5"]] = Field(None, description="Skin type")

