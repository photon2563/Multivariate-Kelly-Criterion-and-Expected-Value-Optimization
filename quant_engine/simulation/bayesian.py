import numpy as np
from typing import Tuple

class DirichletMultinomialModel:
    """
    Implements a Dirichlet-Multinomial Bayesian model for simulating sports outcomes.
    """
    def __init__(self, prior: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        Initializes the model with a Dirichlet prior.
        Default is a uniform prior D(1,1,1) representing Home, Draw, Away.
        """
        self.prior = np.array(prior, dtype=float)

    def get_posterior(self, observed_counts: np.ndarray) -> np.ndarray:
        """
        Calculates the exact Bayesian posterior Dirichlet parameters 
        given empirical Multinomial count data.
        """
        return self.prior + observed_counts

    def sample_probabilities(self, posterior_alpha: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """
        Samples true probability tensors from the Dirichlet posterior.
        """
        return np.random.dirichlet(posterior_alpha, size=n_samples)

def linear_opinion_pool(
    prob_a: np.ndarray, 
    prob_b: np.ndarray, 
    weight_a: float = 0.5
) -> np.ndarray:
    """
    Combines two probability distributions using a linear opinion pool.
    Useful for combining a Home team's posterior predictive with an Away team's.
    """
    weight_b = 1.0 - weight_a
    combined = (weight_a * prob_a) + (weight_b * prob_b)
    # Ensure it sums perfectly to 1
    return combined / np.sum(combined)
