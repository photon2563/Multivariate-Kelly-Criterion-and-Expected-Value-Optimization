import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre
import logging

logger = logging.getLogger(__name__)

# Precompute Laguerre roots and weights for fast integration
# 50 points is generally more than enough for highly accurate convergence
LAGUERRE_NODES, LAGUERRE_WEIGHTS = roots_laguerre(50)

def optimize_multivariate_kelly(
    probabilities: np.ndarray,
    odds: np.ndarray,
    fractional_scalar: float = 1.0,
    max_total_allocation: float = 1.0
) -> np.ndarray:
    """
    Solves the Multivariate Kelly Optimization problem utilizing Frullani's Integral Identity.
    This reduces the computational complexity from O(2^N) to O(N).
    
    Args:
        probabilities: Array of true win probabilities [p_1, p_2, ..., p_N]
        odds: Array of net decimal odds [b_1, b_2, ..., b_N]
        fractional_scalar: Scalar to dampen the Kelly fraction (e.g., 0.5 for Half Kelly)
        max_total_allocation: Maximum allowed sum of fractions (leverage constraint).
        
    Returns:
        Array of optimal fractional allocations for each wager.
    """
    n_bets = len(probabilities)
    
    if n_bets == 0:
        return np.array([])
        
    probabilities = np.asarray(probabilities)
    odds = np.asarray(odds)
    
    # Filter out negative EV bets before optimization for speed
    evs = probabilities * odds - (1.0 - probabilities)
    if np.all(evs <= 0):
        return np.zeros(n_bets)
        
    def expected_log_wealth_negative(ell: np.ndarray) -> float:
        """
        Calculates the negative expected log wealth using Gauss-Laguerre quadrature
        based on Frullani's Identity and Laplace transform factorizations.
        We return negative because scipy.optimize minimizes.
        """
        l_0 = 1.0 - np.sum(ell)
        
        # We integrate: int_0^inf (1/t) * e^{-t} * (1 - E[e^{-tX'}]) dt
        # Using Laguerre quadrature: int_0^inf e^{-t} f(t) dt approx sum(w_j * f(t_j))
        # Here f(t) = (1 - e^{t} * E[e^{-tX}]) / t
        # Wait, standard Frullani for expected log X:
        # E[log X] = int_0^inf (e^-t - E[e^{-tX}]) / t dt
        # With Laguerre: sum_j w_j * [ (1 - e^{t_j} * Q(t_j)) / t_j ]
        # where Q(t) = E[e^{-tX}]
        
        expected_log = 0.0
        
        for t, w in zip(LAGUERRE_NODES, LAGUERRE_WEIGHTS):
            # Calculate Q(t) = E[e^{-tX}]
            # Q(t) = e^{-t*l_0} * prod_{i=1}^N ( (1 - p_i) + p_i * e^{-t * ell_i * (b_i + 1)} )
            
            # The base capital multiplier is 1.0
            # X = l_0 + sum c_i I_i where c_i = ell_i(b_i + 1)
            # Actually, X = 1 - sum ell_i + sum I_i ell_i(1+b_i)
            term1 = np.exp(-t * l_0)
            
            # Individual bet transform products
            exponents = -t * ell * (odds + 1.0)
            products = (1.0 - probabilities) + probabilities * np.exp(exponents)
            
            q_t = term1 * np.prod(products)
            
            # f(t) = (1 - e^t * Q(t)) / t
            # Because w_j already absorbs e^-t, the integral of g(t)e^-t is sum(w_j * g(t_j))
            # g(t) = (e^-t - Q(t)) / t * e^t = (1 - Q(t)e^t) / t
            
            g_t = (1.0 - q_t * np.exp(t)) / t
            expected_log += w * g_t
            
        return -expected_log

    # Initial guess: small uniform fractions
    initial_guess = np.full(n_bets, 0.001)
    
    # Constraints: sum(ell_i) <= max_total_allocation
    constraints = ({
        'type': 'ineq',
        'fun': lambda ell: max_total_allocation - np.sum(ell)
    })
    
    # Bounds: 0 <= ell_i <= 1
    bounds = tuple((0.0, 1.0) for _ in range(n_bets))
    
    # Optimize utilizing SLSQP
    result = minimize(
        expected_log_wealth_negative,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': False}
    )
    
    if not result.success:
        logger.warning(f"Multivariate Kelly optimization failed to converge: {result.message}")
        return np.zeros(n_bets)
        
    optimal_fractions = result.x
    
    # Apply fractional Kelly scalar and clean up precision errors
    optimal_fractions = optimal_fractions * fractional_scalar
    optimal_fractions[optimal_fractions < 1e-5] = 0.0
    
    return optimal_fractions
