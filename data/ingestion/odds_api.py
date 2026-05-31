import httpx
import asyncio
import logging
from typing import Dict, Any, List, Optional
import os

logger = logging.getLogger(__name__)

class OddsAPIClient:
    """
    Asynchronous client for The Odds API.
    Handles rate limiting, pagination, and fetching live market data.
    """
    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        if not self.api_key:
            logger.warning("ODDS_API_KEY is not set. Data ingestion will fail if not using mocked data.")

    async def get_sports(self) -> List[Dict[str, Any]]:
        """
        Fetches a list of in-season sports.
        """
        url = f"{self.BASE_URL}/sports"
        params = {"apiKey": self.api_key}
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    async def get_odds(
        self,
        sport_key: str = "soccer_epl",
        regions: str = "eu,uk",
        markets: str = "h2h",
        odds_format: str = "decimal"
    ) -> List[Dict[str, Any]]:
        """
        Fetches odds for a specific sport.
        """
        url = f"{self.BASE_URL}/sports/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                # Check for rate limits
                remaining = response.headers.get("x-requests-remaining")
                if remaining:
                    logger.info(f"Odds API Requests Remaining: {remaining}")
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error occurred: {e}")
                raise
            except Exception as e:
                logger.error(f"An error occurred during API fetch: {e}")
                raise
