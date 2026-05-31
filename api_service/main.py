import numpy as np
from fastapi import FastAPI, HTTPException
import uvicorn
from api_service.schemas.market_data import SABRCalibrationRequest, SABRCalibrationResponse
from quant_engine.models.sabr import SABRCalibrator

app = FastAPI(title="Advanced Quant Infrastructure API", 
              description="High-performance derivatives pricing and calibration microservices")

@app.post("/api/v1/calibrate_sabr", response_model=SABRCalibrationResponse)
async def calibrate_sabr_endpoint(request: SABRCalibrationRequest):
    """
    Executes the two-stage Differential Evolution + SLSQP SABR calibration routine.
    """
    try:
        strikes = np.array(request.strikes)
        market_vols = np.array(request.market_vols)
        weights = np.array(request.weights) if request.weights else None
        
        calibrator = SABRCalibrator(
            f=request.forward_rate,
            t=request.time_to_maturity,
            strikes=strikes,
            market_vols=market_vols,
            weights=weights,
            beta=request.beta,
            shift=request.shift
        )
        
        results = calibrator.calibrate()
        
        return SABRCalibrationResponse(
            alpha=results['alpha'],
            beta=results['beta'],
            rho=results['rho'],
            nu=results['nu'],
            shift=results['shift'],
            mse=results['mse'],
            status="SUCCESS"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Start the ASGI server with Uvicorn
    # In production, use Gunicorn with Uvicorn workers
    uvicorn.run(app, host="0.0.0.0", port=8000)
