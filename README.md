# Advanced Quantitative Infrastructure: SABR & Merton Jump-Diffusion

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0%2B-00a393.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.27.0%2B-FF4B4B.svg)
![Numba](https://img.shields.io/badge/Numba-JIT_Optimized-blueviolet.svg)
![Quant](https://img.shields.io/badge/Domain-Quantitative_Finance-gold.svg)

## 1. Executive Summary & Architectural Paradigm
The foundational Black-Scholes equation assumes that the underlying asset follows a continuous geometric Brownian motion, defined by a constant expected rate of return (drift) and a strictly constant diffusion coefficient (volatility). 1  The model further assumes continuous trading, infinitely divisible assets, frictionless markets with zero transaction costs, and deterministic interest rates. 1  Under this restrictive paradigm, the terminal distribution of the underlying asset is perfectly log-normal, and the implied volatility—the singular unobservable parameter within the Black-Scholes analytical formula—should theoretically remain perfectly flat and identical regardless of the option's specific strike price (moneyness) or time to expiration.

This repository contains an elite, production-grade quantitative infrastructure tailored for high-frequency derivatives pricing and multi-curve volatility surface calibration. Engineered specifically for the rigorous standards of tier-1 proprietary trading firms and quantitative hedge funds, this system systematically addresses the mathematical breakdown of the classical Black-Scholes-Merton model.

The architecture is built upon a dual-pillar mathematical framework:
1. **The Stochastic Alpha Beta Rho (SABR) Model**: Designed for the complex term structures of interest rate derivatives (and adapted for equities).
2. **The Merton Jump-Diffusion (MJD) Model**: Engineered for equity and foreign exchange markets characterized by sudden macroeconomic discontinuities and heavy-tailed distributions.

By bridging extremely advanced stochastic calculus with ultra-low latency enterprise software engineering (FastAPI microservices, Numba JIT compilation, and Streamlit interactive dashboards), this project demonstrates a complete end-to-end quant lifecycle—from raw theoretical mathematics to front-office deployment.

---

## 2. Theoretical Foundations: Beyond Black-Scholes

The classical Black-Scholes equation assumes geometric Brownian motion with strictly constant volatility. Post-1987, empirical market observations definitively shattered this assumption, revealing a pronounced "volatility smile" and "skew." Because Black-Scholes assumes log-normal terminal distributions, it systemically underprices out-of-the-money (OTM) options—the exact instruments used for institutional tail-risk hedging.

To extract a continuous 3D volatility surface from discrete market data without introducing static arbitrage (Calendar Spread and Butterfly Spread arbitrage), advanced local and stochastic volatility models are strictly mandated. 

### 2.1 The SABR Stochastic Volatility Model
The SABR model resolves the contradictions of local volatility by modeling volatility itself as a stochastic process governed by Brownian motion, correlated ($\rho$) to the underlying asset. 
*   **Alpha ($\alpha$)**: Controls the overall level of the volatility surface.
*   **Beta ($\beta$)**: Constant Elasticity of Variance (CEV). Fixed to 0.5 for rates, 1.0 for equities.
*   **Rho ($\rho$)**: Controls the asymmetric skew.
*   **Nu ($\nu$)**: The vol-of-vol, controlling the convexity or "smile" extreme wings.

### 2.2 The Merton Jump-Diffusion (MJD) Model
To capture discontinuous price jumps (e.g., earnings gaps, central bank shocks), the MJD model modifies geometric Brownian motion by injecting an independent compound Poisson jump process. This structurally forces the simulated asset return distributions to exhibit the heavy tails and high kurtosis perfectly matching empirical market realities.

---

## 3. Phase-by-Phase Implementation Architecture

This repository is meticulously structured into six core developmental phases.

### Phase 1: Repository Architecture & Tech Stack Setup
The foundation of the infrastructure is built on Python 3.11+, leveraging `NumPy` and `SciPy` for heavy linear algebra. To bridge the gap between academic research and trading floor latency, `Numba` is heavily utilized to compile Python directly into LLVM machine code. The project is strictly domain-driven:
*   `quant_engine/`: Core mathematical C-bound models.
*   `api_service/`: Async REST microservice boundaries.
*   `front_office/`: Client-side UI and Excel integrations.

### Phase 2: Pristine Multi-Curve Bootstrapping
Before stochastic models can run, risk-free discounting must be flawless. Post-2008, discounting and forward projection must be decoupled.
*   **Monotone Convex Spline (Hagan-West, 2006)**: Implemented in `quant_engine/math/interpolation.py`. Instead of naive cubic splines which cause negative forward rate oscillations, this highly restrictive algorithm interpolates discrete forward rates ensuring strictly positive, monotonically increasing discount factors and perfect local shock isolation.

### Phase 3: Ultra-Low Latency SABR Calibration
Implemented in `quant_engine/models/sabr.py`.
*   **Hagan Lognormal Asymptotic Expansion**: Utilized to bypass computationally explosive Monte Carlo simulations during real-time calibration.
*   **JIT-Compiled Least Squares**: The objective function is extracted and decorated with `@njit(cache=True)`.
*   **Two-Stage Optimization**: Deploys a **Differential Evolution (DE)** metaheuristic algorithm to aggressively explore the non-convex parameter space and find the global basin of attraction, instantly handing off to a **Sequential Least Squares Programming (SLSQP)** optimizer for boundary-constrained micro-polishing.

### Phase 4: MJD & Fourier Pricing Engines
Implemented in `quant_engine/models/merton_jd.py`.
*   **Carr-Madan (1999) Fast Fourier Transform (FFT)**: Because the transition densities of jump-diffusions are mathematically intractable in the spatial domain, we map the MJD characteristic function into the complex Fourier domain.
*   By applying a dampening factor ($\alpha$), the non-square-integrable call payoff is regularized, allowing the implementation to execute $O(N \log N)$ simultaneous pricing for entire option chains instantly across thousands of strikes.

### Phase 5: Vectorized Monte Carlo Benchmarking
Implemented in `quant_engine/math/monte_carlo.py`.
*   No analytical model reaches production without stochastic verification. We developed a highly parallelized Euler-Maruyama path generator.
*   **Variance Reduction**: Integrates **Antithetic Variates** (negative correlated paths) and **Black-Scholes Control Variates** to mathematically crush statistical standard errors, allowing for precise benchmarking of the FFT outputs.

### Phase 6: Enterprise Deployment & Front-Office UI
To serve the quantitative math to the actual trading desk:
*   **FastAPI Backend**: A highly concurrent REST API (`api_service/main.py`) protected by `Pydantic` schemas, ready for Docker/Kubernetes deployment.
*   **Elite Streamlit Dashboard**: (`front_office/streamlit/app.py`). A highly customized, glassmorphism-styled UI that fetches live market option chains via `yfinance`, dynamically cleans the liquidity profile, and instantly pipes the data through the JIT SABR calibrator. It utilizes `Plotly` to render the 3D continuous volatility smile interactively.

---

## 4. Execution & Usage

### 4.1 Environment Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd advanced_quant_infrastructure

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install highly optimized dependencies
pip install -r requirements.txt
```

### 4.2 Running the Elite Front-Office Dashboard
To experience the JIT-compiled optimization engines against live market data:
```bash
streamlit run front_office/streamlit/app.py
```
Navigate to `http://localhost:8501`. 
*   **Tab 1**: Input a ticker (e.g., SPY) to witness real-time SABR calibration.
*   **Tab 2**: Interact with the Merton Jump-Diffusion FFT pricing engine.

### 4.3 Running the FastAPI Microservice
For backend integration (e.g., Excel VBA):
```bash
uvicorn api_service.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 5. References & Academic Literature
This infrastructure rigidly adheres to the mathematical formulations outlined in elite quantitative literature:
1.  **Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002).** *Managing Smile Risk.* Wilmott Magazine. (Foundational SABR Expansion).
2.  **Merton, R. C. (1976).** *Option Pricing when Underlying Stock Returns are Discontinuous.* Journal of Financial Economics. (Jump-Diffusion Theory).
3.  **Carr, P., & Madan, D. (1999).** *Option Valuation Using the Fast Fourier Transform.* Journal of Computational Finance. (Fourier Density Inversion).
4.  **Hagan, P. S., & West, G. (2006).** *Interpolation Methods for Curve Construction.* Applied Mathematical Finance. (Monotone Convex Splines).
5.  **Gatheral, J. (2006).** *The Volatility Surface: A Practitioner's Guide.* John Wiley & Sons.

---
