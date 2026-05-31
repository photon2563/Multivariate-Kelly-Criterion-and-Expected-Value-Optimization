def single_asset_kelly(p: float, b: float, fractional_scalar: float = 1.0) -> float:
    """
    Calculates the exact discrete-time Kelly optimal fraction for a single binary proposition.
    
    Args:
        p: True probability of winning (0 < p < 1).
        b: Net decimal odds (profit from a winning unit bet).
        fractional_scalar: Multiplier for Fractional Kelly (e.g., 0.5 for Half Kelly).
        
    Returns:
        The fraction of the bankroll to wager. Bounded at 0 (no negative bets).
    """
    q = 1.0 - p
    f_star = (b * p - q) / b
    return max(0.0, f_star * fractional_scalar)

def merton_kelly_approximation(mu: float, sigma_squared: float, fractional_scalar: float = 1.0) -> float:
    """
    Continuous-time Merton approximation of the Kelly fraction.
    Provides a close approximation when the edge (mu) is small relative to variance.
    
    Args:
        mu: Expected return (edge).
        sigma_squared: Variance of the return.
        fractional_scalar: Multiplier for Fractional Kelly.
        
    Returns:
        The fraction of the bankroll to wager.
    """
    if sigma_squared == 0:
        return 0.0
    f_star = mu / (mu + (sigma_squared / (1.0 + mu)))
    return max(0.0, f_star * fractional_scalar)

def calculate_variance(p: float, b: float) -> float:
    """
    Calculates the variance of a single wager's return.
    """
    return p * (1.0 - p) * ((1.0 + b) ** 2)
