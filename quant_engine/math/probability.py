import numpy as np
from typing import Union, List

def implied_probability(odds: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Converts decimal odds to raw implied probabilities (inverse odds).
    Does NOT remove bookmaker margin.
    """
    return 1.0 / np.array(odds, dtype=float)

def overround(odds: Union[List[float], np.ndarray]) -> float:
    """
    Calculates the booksum (overround / margin) of a set of odds.
    E.g. A booksum of 1.05 implies a 5% margin.
    """
    return float(np.sum(implied_probability(odds)))

def basic_margin_removal(odds: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Applies the basic multiplicative margin removal.
    Assumes margin is applied uniformly across all outcomes.
    Retains favorite-longshot bias.
    """
    pi = implied_probability(odds)
    booksum = np.sum(pi)
    return pi / booksum

def calculate_expected_value(true_probability: float, decimal_odds: float) -> float:
    """
    Calculates the expected value (EV) of a wager.
    EV = (True Probability * Decimal Odds) - 1
    
    Returns EV as a decimal (e.g., 0.05 means 5% expected return).
    """
    return (true_probability * decimal_odds) - 1.0

def calculate_edge(true_probability: float, decimal_odds: float) -> float:
    """
    Alias for calculate_expected_value.
    """
    return calculate_expected_value(true_probability, decimal_odds)
