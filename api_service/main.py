import os
import numpy as np
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uvicorn

from api_service.schemas.market_data import SABRCalibrationRequest, SABRCalibrationResponse
from quant_engine.models.sabr import SABRCalibrator

app = FastAPI(title="Advanced Quant Infrastructure API", 
              description="High-performance derivatives pricing and calibration microservices")

# Enable CORS for local testing if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_yf_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    })
    return session

@app.get("/api/v1/market_data/expirations/{symbol}")
async def get_expirations(symbol: str):
    try:
        tkr = yf.Ticker(symbol, session=get_yf_session())
        expirations = tkr.options
        if not expirations:
            raise HTTPException(status_code=404, detail="No options found for this ticker.")
        spot_price = float(tkr.history(period="1d")['Close'].iloc[-1])
        return {"spot_price": spot_price, "expirations": list(expirations)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/market_data/options/{symbol}/{expiry}")
async def get_options(symbol: str, expiry: str):
    try:
        tkr = yf.Ticker(symbol, session=get_yf_session())
        chain = tkr.option_chain(expiry)
        
        # Convert to dictionary (records format)
        calls = chain.calls.to_dict(orient="records")
        puts = chain.puts.to_dict(orient="records")
        
        return {"calls": calls, "puts": puts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

# Mount the frontend web directory
# The html=True flag automatically serves index.html when root is accessed
web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "front_office", "web")
if not os.path.exists(web_dir):
    os.makedirs(web_dir)
app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
