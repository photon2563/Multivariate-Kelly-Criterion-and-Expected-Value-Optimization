import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Union

def calculate_shin_probabilities(odds: Union[List[float], np.ndarray], max_iterations: int = 1000, tol: float = 1e-12) -> Tuple[np.ndarray, float]:
    """
    Calculates true implied probabilities using H.S. Shin's method of insider estimation.
    This neutralizes the favorite-longshot bias present in bookmaker odds.

    Args:
        odds: An array or list of decimal odds for a single event (e.g., [2.0, 3.2, 3.6])
        max_iterations: Maximum iterations for the numerical solver.
        tol: Convergence tolerance for the sum of probabilities.

    Returns:
        A tuple containing:
            - A numpy array of the true probabilities (summing to 1.0).
            - The estimated proportion of insider trading (z).
    """
    odds = np.array(odds, dtype=float)
    if np.any(odds <= 1.0):
        raise ValueError("Decimal odds must be strictly greater than 1.0")

    pi = 1.0 / odds
    booksum = np.sum(pi)

    if booksum <= 1.0:
        # No margin exists; return raw normalized probabilities (z=0)
        return pi / booksum, 0.0

    def objective(z: float) -> float:
        """
        Objective function to find z where the sum of true probabilities equals 1.
        p_i = (sqrt(z^2 + 4 * (1-z) * (pi_i / booksum)) - z) / (2 * (1-z))
        """
        if z >= 1.0 or z < 0.0:
            return 1e9  # Penalize out of bounds

        term = z**2 + 4 * (1 - z) * (pi / booksum)
        # Ensure non-negative term inside sqrt
        term = np.maximum(term, 0)
        
        p = (np.sqrt(term) - z) / (2 * (1 - z))
        return (np.sum(p) - 1.0)**2

    # Initial guess for z is 0.01 (1% insider trading)
    initial_guess = np.array([0.01])
    bounds = [(0.0, 0.9999)]

    # Use SLSQP for bounded optimization
    result = minimize(objective, initial_guess, bounds=bounds, method='SLSQP', tol=tol, options={'maxiter': max_iterations})

    if not result.success and result.fun > tol:
        # Fallback to basic multiplicative margin removal if Shin fails to converge
        return pi / booksum, 0.0

    z_opt = result.x[0]

    if z_opt >= 0.9999:
        # Degenerate case, return basic normalization
        return pi / booksum, 0.0

    # Calculate final probabilities using optimal z
    term = z_opt**2 + 4 * (1 - z_opt) * (pi / booksum)
    p_true = (np.sqrt(term) - z_opt) / (2 * (1 - z_opt))
    
    # Final normalization step to guarantee sum to exactly 1.0 due to float precision
    p_true = p_true / np.sum(p_true)

    return p_true, float(z_opt)
