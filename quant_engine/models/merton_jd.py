import numpy as np
from scipy.fft import fft
from typing import Tuple

class MertonJumpDiffusion:
    """
    Merton Jump-Diffusion (1976) model representation.
    """
    def __init__(self, s0: float, r: float, q: float, sigma: float, lam: float, mu_j: float, delta: float):
        self.s0 = s0
        self.r = r      # Risk-free rate
        self.q = q      # Continuous dividend yield
        self.sigma = sigma # Continuous Brownian volatility
        self.lam = lam    # Jump intensity (expected jumps per year)
        self.mu_j = mu_j  # Mean of the log-jump size
        self.delta = delta # Volatility of the log-jump size

    def characteristic_function(self, u: complex, t: float) -> complex:
        """
        Analytically computes the characteristic function of the log-asset price
        under the Merton Jump-Diffusion model.
        """
        # Risk-neutral drift correction (compensator) for the jump process
        omega = -self.lam * (np.exp(self.mu_j + 0.5 * self.delta**2) - 1.0)
        
        # Drift of the continuous part
        mu_c = self.r - self.q - 0.5 * self.sigma**2 + omega
        
        # The characteristic exponent Psi(u)
        psi_u = 1j * u * mu_c - 0.5 * (self.sigma**2) * (u**2) + \
                self.lam * (np.exp(1j * u * self.mu_j - 0.5 * (self.delta**2) * (u**2)) - 1.0)
                
        return np.exp(psi_u * t)

class CarrMadanFFT:
    """
    Carr-Madan (1999) Fast Fourier Transform Option Pricing Engine.
    """
    def __init__(self, model, n_power: int = 12, eta: float = 0.25, alpha: float = 1.5):
        self.model = model
        self.N = 2 ** n_power # Number of grid points (must be power of 2 for FFT)
        self.eta = eta        # Integration grid spacing (frequency domain)
        self.alpha = alpha    # Dampening factor

    def price_european_calls(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prices a massive chain of European call options simultaneously using FFT.
        Returns: (strikes, call_prices)
        """
        # Grid spacing for log-strike output based on Nyquist relation
        lambda_grid = (2 * np.pi) / (self.N * self.eta)
        
        # Initialize grids
        v_j = np.arange(self.N) * self.eta
        
        # Simpson's rule weights for integration
        weights = np.ones(self.N)
        weights[1::2] = 4.0
        weights[2::2] = 2.0
        weights[0] = 1.0
        weights[-1] = 1.0
        weights = (self.eta / 3.0) * weights
        
        # Calculate the damped characteristic function
        # We need the characteristic function of the log-return: ln(S_T / S_0)
        # So we evaluate at u = v_j - i(alpha + 1)
        u_shifted = v_j - 1j * (self.alpha + 1.0)
        phi = self.model.characteristic_function(u_shifted, t)
        
        # Carr-Madan damped transform
        denominator = (self.alpha + 1j * v_j) * (self.alpha + 1.0 + 1j * v_j)
        zeta = np.exp(-self.model.r * t) * phi / denominator
        
        # Minimum log-strike b
        b = 0.5 * self.N * lambda_grid
        
        # FFT Array construction
        x_j = np.exp(1j * b * v_j) * zeta * weights
        
        # Execute the Fast Fourier Transform
        y_k = fft(x_j).real
        
        # Recover actual strikes and option prices
        k_indices = np.arange(self.N)
        log_strikes = -b + k_indices * lambda_grid
        strikes = self.model.s0 * np.exp(log_strikes)
        
        call_prices = (np.exp(-self.alpha * log_strikes) / np.pi) * y_k
        
        # Note: In a true production environment, we would use spline interpolation 
        # here to map the FFT output grid exactly to the market requested strikes.
        return strikes, call_prices
