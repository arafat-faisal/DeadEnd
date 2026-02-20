"""
News Sentiment Guard for Elite Trading System

Uses CryptoPanic API to gauge market sentiment and provides a score from -1.0 to 1.0.
"""

import requests
import os
from typing import Dict, Any
from utils.logger import get_logger

logger = get_logger('sentiment')

class SentimentGuard:
    def __init__(self):
        # The key isn't provided, should be in .env. We use a placeholder logic internally if missing.
        self.api_key = os.environ.get('CRYPTOPANIC_API_KEY', '')
        self.base_url = "https://cryptopanic.com/api/v1/posts/"
        
    def get_sentiment_score(self, currency: str = None) -> float:
        """
        Fetch recent news from CryptoPanic and calculate a sentiment score.
        Score ranges from -1.0 (extremely bearish/negative) to 1.0 (extremely bullish/positive).
        """
        if not self.api_key:
            logger.warning("No CryptoPanic API key found. Returning neutral sentiment (0.0).")
            return 0.0
            
        params = {
            'auth_token': self.api_key,
            'filter': 'important',
            'kind': 'news'
        }
        
        if currency:
            params['currencies'] = currency

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to fetch sentiment: HTTP {response.status_code}")
                return 0.0
                
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                return 0.0
                
            positive_votes = 0
            negative_votes = 0
            
            for post in results[:10]: # Look at latest 10 important posts
                votes = post.get('votes', {})
                positive_votes += votes.get('positive', 0) + votes.get('bullish', 0) + votes.get('important', 0)
                negative_votes += votes.get('negative', 0) + votes.get('bearish', 0) + votes.get('toxic', 0)
                
            total_votes = positive_votes + negative_votes
            if total_votes == 0:
                return 0.0
                
            # Score calculation
            score = (positive_votes - negative_votes) / total_votes
            logger.debug(f"Sentiment score calculated: {score:.2f} (Pos: {positive_votes}, Neg: {negative_votes})")
            return max(-1.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error fetching sentiment: {e}")
            return 0.0
