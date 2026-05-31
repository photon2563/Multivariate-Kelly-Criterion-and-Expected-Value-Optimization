import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import njit

@njit(cache=True)
def hagan_lognormal_vol(f: float, k: float, t: float, alpha: float, beta: float, rho: float, nu: float, shift: float = 0.0) -> float:
    """
    Computes the Black-Scholes implied volatility using the Hagan et al. (2002) 
    asymptotic expansion for the Shifted SABR model.
    Uses numba JIT compilation for extreme performance in calibration loops.
    """
    f = f + shift
    k = k + shift

    # Prevent division by zero or negative rates (if shift is insufficient)
    if f <= 0 or k <= 0:
        return 0.0
    
    if f == k: # ATM case
        term1 = alpha / (f ** (1 - beta))
        term2 = 1.0 + (((1 - beta) ** 2) / 24.0 * (alpha ** 2) / (f ** (2 - 2 * beta)) + 
                       (rho * beta * nu * alpha) / (4.0 * (f ** (1 - beta))) + 
                       ((2 - 3 * (rho ** 2)) / 24.0) * (nu ** 2)) * t
        return term1 * term2

    # General OTM/ITM case
    log_fk = np.log(f / k)
    f_k_beta = (f * k) ** ((1 - beta) / 2.0)
    
    z = (nu / alpha) * f_k_beta * log_fk
    
    # Handle the degenerate case where rho and z make the denominator of chi 0
    numerator_chi = np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho
    if numerator_chi <= 0:
         return 0.0
         
    chi_z = np.log(numerator_chi / (1 - rho))
    
    term1 = alpha / (f_k_beta * (1 + (((1 - beta) ** 2) / 24.0) * (log_fk ** 2) + 
                                 (((1 - beta) ** 4) / 1920.0) * (log_fk ** 4)))
    
    # If z is extremely small, use Taylor expansion for z/chi(z) to avoid 0/0
    if abs(z) < 1e-7:
        term2 = 1.0
    else:
        term2 = z / chi_z
        
    term3 = 1.0 + (((1 - beta) ** 2) / 24.0 * (alpha ** 2) / ((f * k) ** (1 - beta)) + 
                   (rho * beta * nu * alpha) / (4.0 * f_k_beta) + 
                   ((2 - 3 * (rho ** 2)) / 24.0) * (nu ** 2)) * t
                   
    return term1 * term2 * term3

@njit(cache=True)
def sabr_objective_function(params: np.ndarray, f: float, t: float, strikes: np.ndarray, 
                            market_vols: np.ndarray, weights: np.ndarray, beta: float, shift: float) -> float:
    """Highly optimized Numba JIT objective function for DE/SLSQP solvers."""
    alpha, rho, nu = params
    error = 0.0
    for i in range(len(strikes)):
        model_vol = hagan_lognormal_vol(f, strikes[i], t, alpha, beta, rho, nu, shift)
        error += weights[i] * (model_vol - market_vols[i]) ** 2
    return error

class SABRCalibrator:
    """
    Production-grade SABR Calibration Engine.
    Uses Differential Evolution for global minimum discovery, followed by SLSQP for polishing.
    """
    def __init__(self, f: float, t: float, strikes: np.ndarray, market_vols: np.ndarray, 
                 weights: np.ndarray = None, beta: float = 0.5, shift: float = 0.0):
        self.f = f
        self.t = t
        self.strikes = strikes
        self.market_vols = market_vols
        self.beta = beta
        self.shift = shift
        self.weights = weights if weights is not None else np.ones_like(strikes)

    def calibrate(self) -> dict:
        """Executes the two-stage calibration routine."""
        # Parameter Bounds: alpha > 0, -1 <= rho <= 1, nu > 0
        bounds = [(1e-6, 2.0), (-0.999, 0.999), (1e-6, 5.0)]
        
        # Arguments for the objective function
        args = (self.f, self.t, self.strikes, self.market_vols, self.weights, self.beta, self.shift)
        
        # Stage 1: Global Search (Differential Evolution)
        # We use a smaller popsize for speed in production, assuming standard smile shapes
        de_result = differential_evolution(sabr_objective_function, bounds, args=args,
                                           strategy='best1bin', popsize=10, tol=1e-4)
        
        # Stage 2: Local Polishing (SLSQP)
        slsqp_result = minimize(sabr_objective_function, de_result.x, args=args,
                                method='SLSQP', bounds=bounds, tol=1e-8)
        
        if not slsqp_result.success:
            # Fallback to DE result if SLSQP fails (rare but possible in extremely flat markets)
            optimal_params = de_result.x
        else:
            optimal_params = slsqp_result.x
            
        return {
            'alpha': optimal_params[0],
            'beta': self.beta,
            'rho': optimal_params[1],
            'nu': optimal_params[2],
            'shift': self.shift,
            'mse': slsqp_result.fun
        }
