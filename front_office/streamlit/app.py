import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
import sys
import os
from datetime import datetime

# Add the root directory to path to import quant_engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from quant_engine.models.sabr import SABRCalibrator, hagan_lognormal_vol

st.set_page_config(page_title="Advanced Quant Infrastructure", layout="wide", initial_sidebar_state="expanded")

# Injecting Elite Custom CSS for Premium Design Aesthetics
st.markdown("""
    <style>
    /* Global Background and Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Glassmorphism for sidebar */
    [data-testid="stSidebar"] {
        background: rgba(22, 27, 34, 0.7) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Stylized Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        background: -webkit-linear-gradient(45deg, #58a6ff, #8a2be2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Buttons with micro-animations */
    .stButton>button {
        background: linear-gradient(135deg, #1f6feb 0%, #3fb950 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(31, 111, 235, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(31, 111, 235, 0.5);
    }
    
    /* Headers */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #ffffff, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Advanced Quantitative Infrastructure: SABR Volatility Calibration")
st.markdown("This elite dashboard demonstrates production-grade SABR model calibration using ultra-low latency JIT-compiled optimizers and real-time options data fetched via `yfinance`.")

# Sidebar Controls
st.sidebar.header("Market Data Parameters")
ticker_symbol = st.sidebar.text_input("Ticker Symbol", value="SPY")

import requests

def get_yf_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    })
    return session

@st.cache_data(ttl=300)
def fetch_spot_and_expirations(symbol):
    try:
        tkr = yf.Ticker(symbol, session=get_yf_session())
        expirations = tkr.options
        if not expirations:
            return None, None, "No options found for this ticker."
        spot_price = tkr.history(period="1d")['Close'].iloc[-1]
        return spot_price, list(expirations), None
    except Exception as e:
        return None, None, str(e)

@st.cache_data(ttl=300)
def fetch_option_chain(symbol, expiry):
    tkr = yf.Ticker(symbol, session=get_yf_session())
    return tkr.option_chain(expiry)

spot_price, expirations, error_msg = fetch_spot_and_expirations(ticker_symbol)

if spot_price is None:
    st.error(f"No options data found for ticker {ticker_symbol}.")
    if error_msg:
        st.warning(f"yfinance API Details: {error_msg}")
        st.info("Note: Yahoo Finance occasionally blocks cloud IP addresses (like Streamlit Community Cloud). If this persists, try running the app locally.")
else:
    st.sidebar.metric("Current Spot Price", f"${spot_price:.2f}")
    
    selected_expiry = st.sidebar.selectbox("Select Expiration Date", expirations)
    
    # Calculate time to maturity in years (approximate based on trading days or calendar days)
    expiry_date = datetime.strptime(selected_expiry, '%Y-%m-%d')
    today = datetime.now()
    days_to_expiry = (expiry_date - today).days
    time_to_maturity = max(days_to_expiry / 365.25, 0.001)
    st.sidebar.text(f"Time to Maturity: {time_to_maturity:.4f} years")

    # Fetch Option Chain
    option_chain = fetch_option_chain(ticker_symbol, selected_expiry)
    
    # Focus on Calls for simplicity in this demo, and filter out near-zero implied vols or illiquid strikes
    calls = option_chain.calls
    calls = calls[(calls['impliedVolatility'] > 0.01) & (calls['volume'] > 0)]
    
    # Filter strikes around the money (e.g., +/- 20%)
    lower_bound = spot_price * 0.8
    upper_bound = spot_price * 1.2
    calls = calls[(calls['strike'] >= lower_bound) & (calls['strike'] <= upper_bound)]
    
    if len(calls) < 5:
        st.warning("Not enough liquid options data for this expiration to perform a stable calibration.")
    else:
        strikes = calls['strike'].values
        market_vols = calls['impliedVolatility'].values
        
    tab1, tab2 = st.tabs(["SABR Stochastic Volatility", "Merton Jump-Diffusion (MJD)"])
    
    with tab1:
        st.subheader(f"Market Implied Volatility Smile for {ticker_symbol} (Expiry: {selected_expiry})")
        
        # SABR Calibration Params
        st.markdown("### SABR Configuration")
        # For Equities, beta is typically 1.0 (Lognormal CEV)
        beta = st.number_input("Beta (CEV Exponent)", value=1.0, min_value=0.0, max_value=1.0)
        
        if st.button("Run Highly-Optimized SABR Calibration"):
            with st.spinner("Calibrating SABR using Differential Evolution and SLSQP..."):
                # Initialize Calibrator
                calibrator = SABRCalibrator(
                    f=spot_price,
                    t=time_to_maturity,
                    strikes=strikes,
                    market_vols=market_vols,
                    beta=beta
                )
                results = calibrator.calibrate()
                
            st.success("Calibration Complete!")
            
            # Display Params
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Alpha", f"{results['alpha']:.4f}")
            col2.metric("Beta", f"{results['beta']:.4f}")
            col3.metric("Rho", f"{results['rho']:.4f}")
            col4.metric("Nu (Vol-of-Vol)", f"{results['nu']:.4f}")
            
            # Generate continuous smile curve for plotting
            dense_strikes = np.linspace(min(strikes), max(strikes), 100)
            sabr_vols = [
                hagan_lognormal_vol(spot_price, k, time_to_maturity, 
                                    results['alpha'], results['beta'], 
                                    results['rho'], results['nu']) 
                for k in dense_strikes
            ]
            
            # Plotly Graph
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=strikes, y=market_vols, mode='markers', name='Market Implied Vols (yfinance)'))
            fig.add_trace(go.Scatter(x=dense_strikes, y=sabr_vols, mode='lines', name='Calibrated SABR Smile'))
            fig.update_layout(
                title="SABR Calibration vs Market Data",
                xaxis_title="Strike Price",
                yaxis_title="Implied Volatility",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**Mean Squared Error (MSE):** `{results['mse']:.8f}`")
            
    with tab2:
        st.subheader("Merton Jump-Diffusion (MJD) Pricing Engine")
        st.markdown("Demonstrating the **Carr-Madan FFT** pricing capability over heavy-tailed, discontinuous price distributions.")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        mjd_lambda = col_m1.slider("Jump Intensity (λ)", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
        mjd_mu_j = col_m2.slider("Mean Log-Jump (μ_j)", min_value=-0.5, max_value=0.5, value=-0.1, step=0.05)
        mjd_delta = col_m3.slider("Jump Volatility (δ)", min_value=0.01, max_value=0.5, value=0.15, step=0.01)
        
        if st.button("Price via Carr-Madan Fast Fourier Transform"):
            from quant_engine.models.merton_jd import MertonJumpDiffusion, CarrMadanFFT
            with st.spinner("Executing O(N log N) FFT Integration..."):
                # Approximate typical BS vol from the ATM options
                atm_vol = np.median(market_vols) if len(market_vols) > 0 else 0.2
                
                mjd_model = MertonJumpDiffusion(
                    s0=spot_price, r=0.05, q=0.0, sigma=atm_vol, 
                    lam=mjd_lambda, mu_j=mjd_mu_j, delta=mjd_delta
                )
                fft_pricer = CarrMadanFFT(mjd_model, n_power=12)
                fft_strikes, fft_prices = fft_pricer.price_european_calls(time_to_maturity)
                
                # Filter to relevant strikes around ATM
                valid_indices = (fft_strikes >= lower_bound) & (fft_strikes <= upper_bound)
                plot_strikes = fft_strikes[valid_indices]
                plot_prices = fft_prices[valid_indices]
                
            st.success("FFT Pricing Complete!")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=plot_strikes, y=plot_prices, mode='lines', name='MJD Call Prices', line=dict(color='#00ff88')))
            fig2.add_vline(x=spot_price, line_dash="dash", line_color="white", annotation_text="Spot Price")
            fig2.update_layout(
                title="European Call Option Chain (Merton Jump-Diffusion via FFT)",
                xaxis_title="Strike Price",
                yaxis_title="Option Premium ($)",
                template="plotly_dark"
            )
            st.plotly_chart(fig2, use_container_width=True)
