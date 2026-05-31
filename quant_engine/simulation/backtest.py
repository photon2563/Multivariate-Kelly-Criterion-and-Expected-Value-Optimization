import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from quant_engine.portfolio_optimization.kelly import single_asset_kelly
from quant_engine.portfolio_optimization.multivariate import optimize_multivariate_kelly

logger = logging.getLogger(__name__)

class MonteCarloBacktester:
    """
    Massive Monte Carlo simulation engine comparing Flat Betting, Independent Kelly, 
    and Simultaneous Multivariate Kelly.
    """
    def __init__(
        self,
        n_seasons: int = 1000,
        matches_per_season: int = 380,
        simultaneous_matches: int = 10,
        initial_bankroll: float = 10000.0,
        fractional_kelly: float = 0.35,
        flat_bet_fraction: float = 0.01
    ):
        self.n_seasons = n_seasons
        self.matches_per_season = matches_per_season
        self.simultaneous_matches = simultaneous_matches
        self.initial_bankroll = initial_bankroll
        self.fractional_kelly = fractional_kelly
        self.flat_bet_fraction = flat_bet_fraction

    def generate_synthetic_slate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates a synthetic slate of simultaneous matches with true probabilities and noisy odds.
        Returns: true_probs, decimal_odds, outcomes
        """
        # Generate random true probabilities for Home win (simplified binary outcome for testing)
        true_probs = np.random.uniform(0.3, 0.8, size=self.simultaneous_matches)
        
        # Add a random overround/margin (2% to 6%)
        margins = np.random.uniform(1.02, 1.06, size=self.simultaneous_matches)
        
        # Inject random Gaussian noise to create edges
        noise = np.random.normal(0, 0.05, size=self.simultaneous_matches)
        implied_probs = np.clip(true_probs + noise, 0.05, 0.95)
        
        # Bookmaker decimal odds (including margin)
        decimal_odds = 1.0 / (implied_probs * margins)
        
        # Simulate actual outcomes (1 for win, 0 for loss)
        random_rolls = np.random.uniform(0, 1, size=self.simultaneous_matches)
        outcomes = (random_rolls <= true_probs).astype(float)
        
        return true_probs, decimal_odds, outcomes

    def run_season(self) -> Dict[str, Any]:
        """Runs a single season simulation."""
        bankrolls = {
            "Flat": self.initial_bankroll,
            "Independent_Kelly": self.initial_bankroll,
            "Multivariate_Kelly": self.initial_bankroll
        }
        
        history = {k: [self.initial_bankroll] for k in bankrolls.keys()}
        
        n_slates = self.matches_per_season // self.simultaneous_matches
        
        for _ in range(n_slates):
            p, b, outcomes = self.generate_synthetic_slate()
            
            # Find positive EV bets
            evs = (p * b) - 1.0
            valid_idx = evs > 0
            
            if not np.any(valid_idx):
                for k in bankrolls:
                    history[k].append(bankrolls[k])
                continue
                
            p_edge = p[valid_idx]
            b_edge = b[valid_idx]
            outcomes_edge = outcomes[valid_idx]
            
            # 1. Flat Betting
            n_bets = len(p_edge)
            flat_wager = bankrolls["Flat"] * self.flat_bet_fraction
            total_flat_wager = flat_wager * n_bets
            if total_flat_wager > bankrolls["Flat"]:
                flat_wager = bankrolls["Flat"] / n_bets
            
            flat_returns = np.sum(flat_wager * (outcomes_edge * b_edge - 1.0))
            bankrolls["Flat"] += flat_returns
            
            # 2. Independent Kelly
            indep_fractions = np.array([
                single_asset_kelly(p_edge[i], b_edge[i] - 1.0, self.fractional_kelly) 
                for i in range(n_bets)
            ])
            # Cap total leverage at 1.0 to prevent immediate ruin
            total_indep_frac = np.sum(indep_fractions)
            if total_indep_frac > 1.0:
                indep_fractions /= total_indep_frac
                
            indep_returns = np.sum(bankrolls["Independent_Kelly"] * indep_fractions * (outcomes_edge * b_edge - 1.0))
            bankrolls["Independent_Kelly"] += indep_returns
            
            # 3. Multivariate Kelly
            multi_fractions = optimize_multivariate_kelly(
                p_edge, b_edge - 1.0, self.fractional_kelly, max_total_allocation=1.0
            )
            multi_returns = np.sum(bankrolls["Multivariate_Kelly"] * multi_fractions * (outcomes_edge * b_edge - 1.0))
            bankrolls["Multivariate_Kelly"] += multi_returns
            
            # Record keeping and bankruptcy checks
            for k in bankrolls:
                if bankrolls[k] <= 0:
                    bankrolls[k] = 0.0
                history[k].append(bankrolls[k])
                
        return history

    def run_simulation(self):
        """Runs massive Monte Carlo across all seasons."""
        logger.info(f"Starting {self.n_seasons}-season Monte Carlo Backtest...")
        results = {"Flat": [], "Independent_Kelly": [], "Multivariate_Kelly": []}
        
        for i in range(self.n_seasons):
            season_history = self.run_season()
            for k in results:
                final_b = season_history[k][-1]
                results[k].append(final_b)
                
        return pd.DataFrame(results)
