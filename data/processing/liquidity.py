import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
import sys
import os

# Add quant_engine to path for relative imports if run as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from quant_engine.math.shin import calculate_shin_probabilities
from quant_engine.math.probability import calculate_expected_value

logger = logging.getLogger(__name__)

# Define sharp bookmakers that act as our source of truth
SHARP_BOOKMAKERS = ["pinnacle", "betfair_ex_eu", "matchbook"]

def find_edges(
    events_data: List[Dict[str, Any]],
    min_edge_threshold: float = 0.03
) -> pd.DataFrame:
    """
    Parses live Odds API data, identifies sharp liquidity, calculates true probabilities via Shin,
    and scans soft bookmakers for positive EV opportunities.
    
    Args:
        events_data: Raw JSON response from Odds API.
        min_edge_threshold: Minimum expected value to flag an opportunity (e.g. 0.03 for 3%).
        
    Returns:
        A pandas DataFrame of identified edges containing event details, bookmaker, outcome, and EV.
    """
    opportunities = []

    for event in events_data:
        event_id = event.get("id")
        sport_title = event.get("sport_title")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        commence_time = event.get("commence_time")
        bookmakers = event.get("bookmakers", [])

        sharp_odds_dict = {}
        soft_bookmakers = []

        # Separate sharp and soft bookmakers
        for bookmaker in bookmakers:
            key = bookmaker.get("key")
            markets = bookmaker.get("markets", [])
            # Focus on h2h (moneyline) for this implementation
            h2h_market = next((m for m in markets if m.get("key") == "h2h"), None)
            
            if not h2h_market:
                continue
                
            outcomes = h2h_market.get("outcomes", [])
            
            if key in SHARP_BOOKMAKERS:
                sharp_odds_dict[key] = outcomes
            else:
                soft_bookmakers.append((key, outcomes))

        if not sharp_odds_dict:
            # No sharp liquidity available to establish true baseline
            continue

        # Prefer Pinnacle if available, else take the first available sharp
        selected_sharp = sharp_odds_dict.get("pinnacle") or list(sharp_odds_dict.values())[0]
        
        # Extract odds and apply Shin's method
        try:
            # Sort outcomes alphabetically by name to ensure consistency
            selected_sharp.sort(key=lambda x: x["name"])
            sharp_odds_values = [outcome["price"] for outcome in selected_sharp]
            outcome_names = [outcome["name"] for outcome in selected_sharp]
            
            true_probs, z_value = calculate_shin_probabilities(sharp_odds_values)
        except Exception as e:
            logger.debug(f"Failed to calculate true probabilities for {event_id}: {e}")
            continue

        true_prob_map = dict(zip(outcome_names, true_probs))

        # Scan soft bookmakers for edges
        for soft_key, soft_outcomes in soft_bookmakers:
            for outcome in soft_outcomes:
                name = outcome["name"]
                soft_odds = outcome["price"]
                
                true_p = true_prob_map.get(name)
                if true_p is None:
                    continue
                    
                ev = calculate_expected_value(true_p, soft_odds)
                
                if ev >= min_edge_threshold:
                    opportunities.append({
                        "event_id": event_id,
                        "sport": sport_title,
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": commence_time,
                        "bookmaker": soft_key,
                        "outcome": name,
                        "true_probability": true_p,
                        "decimal_odds": soft_odds,
                        "expected_value": ev,
                        "insider_z": z_value
                    })

    df = pd.DataFrame(opportunities)
    return df
