from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from quant_engine.portfolio_optimization.multivariate import optimize_multivariate_kelly

router = APIRouter(
    prefix="/kelly",
    tags=["Portfolio Optimization"]
)

class KellyRequest(BaseModel):
    probabilities: List[float] = Field(..., description="List of true win probabilities")
    odds: List[float] = Field(..., description="List of decimal odds")
    fractional_scalar: float = Field(0.35, description="Fractional Kelly dampening scalar")
    max_allocation: float = Field(1.0, description="Maximum total portfolio allocation")

class KellyResponse(BaseModel):
    allocations: List[float]
    total_allocation: float

@router.post("/multivariate", response_model=KellyResponse)
async def calculate_multivariate_kelly(request: KellyRequest):
    """
    Calculates the optimal simultaneous Kelly fractions for a portfolio of independent wagers.
    Uses Frullani's Integral transform for O(N) complexity.
    """
    if len(request.probabilities) != len(request.odds):
        raise HTTPException(status_code=400, detail="Length of probabilities and odds must match.")
        
    try:
        p_array = np.array(request.probabilities)
        b_array = np.array(request.odds) - 1.0 # Convert to net odds
        
        allocations = optimize_multivariate_kelly(
            probabilities=p_array,
            odds=b_array,
            fractional_scalar=request.fractional_scalar,
            max_total_allocation=request.max_allocation
        )
        
        return KellyResponse(
            allocations=allocations.tolist(),
            total_allocation=float(np.sum(allocations))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
