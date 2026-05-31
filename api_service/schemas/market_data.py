from pydantic import BaseModel, Field
from typing import List

class SABRCalibrationRequest(BaseModel):
    forward_rate: float = Field(..., gt=0, description="Current forward rate (e.g. swap rate)")
    time_to_maturity: float = Field(..., gt=0, description="Time to expiry in years")
    strikes: List[float] = Field(..., description="Array of strike prices")
    market_vols: List[float] = Field(..., description="Array of market implied Black volatilities")
    weights: List[float] | None = Field(None, description="Optional array of weights for calibration")
    beta: float = Field(0.5, description="CEV exponent, typically fixed to 0.5 for rates")
    shift: float = Field(0.0, description="Shift parameter for negative rates")

class SABRCalibrationResponse(BaseModel):
    alpha: float
    beta: float
    rho: float
    nu: float
    shift: float
    mse: float
    status: str
