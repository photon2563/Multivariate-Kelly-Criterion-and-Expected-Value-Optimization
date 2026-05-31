import numpy as np
from typing import List, Dict, Optional
from quant_engine.math.interpolation import InterpolationMethod, LogCubicSpline

class YieldCurve:
    """
    Core representation of a Zero-Coupon Yield Curve.
    Used for both risk-free discounting (OIS) and forward rate projection.
    """
    def __init__(self, ref_date: str, interpolation: InterpolationMethod = LogCubicSpline()):
        self.ref_date = ref_date
        self.interpolation = interpolation
        self.nodes_t = np.array([])  # Time to maturity in years
        self.nodes_z = np.array([])  # Zero-coupon discount factors Z(0, t)

    def build_curve(self, tenors: List[float], discount_factors: List[float]):
        """Builds the internal curve representation from bootstrapped nodes."""
        if len(tenors) != len(discount_factors):
            raise ValueError("Tenors and discount factors must have the same length.")
        
        # Sort nodes by tenor
        sorted_indices = np.argsort(tenors)
        self.nodes_t = np.array(tenors)[sorted_indices]
        self.nodes_z = np.array(discount_factors)[sorted_indices]

    def get_discount_factor(self, t: float) -> float:
        """Returns the interpolated discount factor for a given time t."""
        if len(self.nodes_t) == 0:
            raise ValueError("Curve is not built. Please provide nodes.")
        
        # Exact node match
        if t in self.nodes_t:
            idx = np.where(self.nodes_t == t)[0][0]
            return self.nodes_z[idx]
            
        # Interpolate
        return self.interpolation.interpolate(self.nodes_t, self.nodes_z, np.array([t]))[0]

    def get_forward_rate(self, t1: float, t2: float) -> float:
        """
        Calculates the simply-compounded forward rate between t1 and t2.
        F(t1, t2) = (1 / delta_t) * [Z(0, t1) / Z(0, t2) - 1]
        """
        if t1 >= t2:
            raise ValueError("t2 must be strictly greater than t1.")
        
        z1 = self.get_discount_factor(t1)
        z2 = self.get_discount_factor(t2)
        dt = t2 - t1
        
        return (1.0 / dt) * ((z1 / z2) - 1.0)

class DualCurveFramework:
    """
    Represents the post-2008 Dual-Curve framework.
    Decouples discounting (OIS) from forward projection (e.g., 3M/6M LIBOR/SOFR term).
    """
    def __init__(self, discount_curve: YieldCurve):
        self.discount_curve = discount_curve
        self.forward_curves: Dict[str, YieldCurve] = {}

    def add_forward_curve(self, tenor_id: str, curve: YieldCurve):
        """Adds a specific forward projection curve (e.g., '3M', '6M')."""
        self.forward_curves[tenor_id] = curve

    def price_cashflow(self, amount: float, payment_time: float) -> float:
        """Prices a deterministic cashflow using the OIS discount curve."""
        return amount * self.discount_curve.get_discount_factor(payment_time)
