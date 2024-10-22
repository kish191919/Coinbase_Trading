import os
import json
import pandas as pd
import http.client
import logging
import jwt
import time
import secrets
from functools import lru_cache
from datetime import datetime, timedelta
from dotenv import load_dotenv
from cryptography.hazmat.primitives import serialization
from typing import Tuple, Optional, Dict, Any
from urllib.parse import quote

from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.utils import dropna

# Configure pandas display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class Autotrade:
    """
    A class for automated cryptocurrency trading using Coinbase API.
    Handles authentication, data retrieval, and technical analysis.
    """

    def __init__(self):
        # Initialize Autotrade with API credentials and connection settings
        self.api_key, self.api_secret = self._get_api_credentials()
        self.base_url = "api.coinbase.com"
        self.exchange_url = "api.exchange.coinbase.com"

        # Store connection and headers for reuse
        self._cached_conn: Optional[http.client.HTTPSConnection] = None
        self._cached_headers: Optional[Dict[str, str]] = None
        self._headers_timestamp: float = 0
        self.HEADERS_EXPIRY = 110  # JWT token expires in 120 seconds, refresh slightly earlier

    # Load API credentials from environment variables.
    @staticmethod
    def _get_api_credentials() -> Tuple[str, str]:
        load_dotenv()
        api_key = os.getenv("api_key")
        api_secret = os.getenv("api_secret")
        if not api_key or not api_secret:
            raise ValueError("API credentials not found in environment variables")
        return api_key, api_secret

    # Build JWT token for API authentication.
    def _build_jwt(self, resource: str) -> str:
        private_key_bytes = self.api_secret.encode('utf-8')
        private_key = serialization.load_pem_private_key(private_key_bytes, password=None)
        uri = f"GET api.coinbase.com/api/v3/brokerage/{resource}"

        jwt_payload = {
            'sub': self.api_key,
            'iss': "cdp",
            'nbf': int(time.time()),
            'exp': int(time.time()) + 120,
            'uri': uri,
        }

        return jwt.encode(
            jwt_payload,
            private_key,
            algorithm='ES256',
            headers={'kid': self.api_key, 'nonce': secrets.token_hex()},
        )

    # Get or create an HTTP connection with appropriate headers.
    def _get_connection(self, resource: str, base_url: str) -> Tuple[dict, http.client.HTTPSConnection]:
        current_time = time.time()

        # Check if we need to refresh headers
        if (not self._cached_headers or
                not self._cached_conn or
                current_time - self._headers_timestamp > self.HEADERS_EXPIRY):

            if self._cached_conn:
                self._cached_conn.close()

            jwt_token = self._build_jwt(resource)
            self._cached_conn = http.client.HTTPSConnection(base_url)
            self._cached_headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {jwt_token}',
                'User-Agent': 'AutoTrade/1.0'
            }
            self._headers_timestamp = current_time

        return self._cached_headers, self._cached_conn

    # Retrieve historical candle data for a specific product.
    def get_candles(self, product_id: str, granularity: int, start: Optional[str] = None, end: Optional[str] = None) -> \
    Optional[pd.DataFrame]:
        try:
            # Create a new connection specifically for exchange API
            conn = http.client.HTTPSConnection(self.exchange_url, timeout=10)

            # Build query parameters
            params = [f"granularity={granularity}"]
            if start:
                params.append(f"start={start}")
            if end:
                params.append(f"end={end}")

            # Construct query URL
            query = f"/products/{product_id}/candles"
            if params:
                query += "?" + "&".join(params)

            # Set headers with proper user agent
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'AutoTrade/1.0'
            }

            logger.debug(f"Requesting candle data: {query}")

            # Make request with timeout
            conn.request("GET", query, '', headers)
            res = conn.getresponse()

            # Check response status
            if res.status != 200:
                logger.error(f"Failed to get candle data. Status: {res.status}")
                return None

            # Parse response data
            data = res.read()
            candles = json.loads(data.decode("utf-8"))

            if not candles:
                logger.warning(f"No candle data received for {product_id}")
                return None

            # Create DataFrame with proper column names
            df = pd.DataFrame(candles,
                              columns=['time', 'low', 'high', 'open', 'close', 'volume'])

            # Convert timestamp to datetime and sort
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.sort_values('time')

            # Remove any duplicate timestamps
            df = df.drop_duplicates(subset=['time'], keep='last')

            # Set time as index for easier analysis
            df.set_index('time', inplace=True)

            return df

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse candle data: {str(e)}")
            return None
        except http.client.HTTPException as e:
            logger.error(f"HTTP error occurred: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting candle data: {str(e)}")
            return None
        finally:
            conn.close()

    # Validate API response status code.
    @staticmethod
    def _check_response(response: http.client.HTTPResponse) -> None:
        if response.status != 200:
            error_msg = f"API request failed with status code: {response.status}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    # Get available balances for USD and BTC.
    @lru_cache(maxsize=32)
    def get_available_balance(self) -> Tuple[float, float]:
        resource = "accounts"
        headers, conn = self._get_connection(resource, self.base_url)

        try:
            conn.request("GET", "/api/v3/brokerage/accounts", '', headers)
            res = conn.getresponse()
            self._check_response(res)
            data = json.loads(res.read().decode("utf-8"))

            balances = {
                'USD': 0.0,
                'BTC': 0.0
            }

            for account in data.get('accounts', []):
                currency = account.get('currency', '')
                if currency in balances and float(account.get('available_balance', {}).get('value', 0)) > 0:
                    balances[currency] = float(account['available_balance']['value'])

            return balances['USD'], balances['BTC']

        except Exception as e:
            logger.error(f"Error getting balances: {str(e)}")
            raise

    # Add technical analysis indicators to the price data.
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return None

        df = dropna(df.copy())  # Work with a copy to prevent SettingWithCopyWarning

        # Calculate all indicators in one pass
        indicators = {
            'bb': BollingerBands(close=df['close'], window=20, window_dev=2),
            'rsi': RSIIndicator(close=df['close'], window=14),
            'macd': MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9),
        }

        # Add Bollinger Bands
        df['bb_middle'] = indicators['bb'].bollinger_mavg()
        df['bb_upper'] = indicators['bb'].bollinger_hband()
        df['bb_lower'] = indicators['bb'].bollinger_lband()
        df['bb_width'] = indicators['bb'].bollinger_wband()
        df['bb_percent'] = indicators['bb'].bollinger_pband()

        # Add RSI
        df['rsi'] = indicators['rsi'].rsi()

        # Add MACD
        df['macd'] = indicators['macd'].macd()
        df['macd_signal'] = indicators['macd'].macd_signal()
        df['macd_diff'] = indicators['macd'].macd_diff()

        # Add SMAs efficiently
        for window in [10, 20, 60]:
            df[f'sma_{window}'] = SMAIndicator(close=df['close'], window=window).sma_indicator()

        return df

    # Get historical price data for different time periods.
    def get_historical_data(self, product_id: str = "BTC-USD") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        end_time = datetime.utcnow()
        periods = {
            '24h': {'days': 1, 'granularity': 300},  # 5-minute intervals
            '30d': {'days': 30, 'granularity': 86400},  # 1-day intervals
            '90d': {'days': 90, 'granularity': 86400}  # 1-day intervals
        }

        results = []
        for period_info in periods.values():
            start_time = end_time - timedelta(days=period_info['days'])
            start_str = start_time.isoformat().split('+')[0] + 'Z'
            end_str = end_time.isoformat().split('+')[0] + 'Z'

            df = self.get_candles(
                product_id=product_id,
                granularity=period_info['granularity'],
                start=quote(start_str),
                end=quote(end_str)
            )
            results.append(df)

        return tuple(results)

    # Get technical analysis for different time periods.
    def get_analysis(self, product_id: str = "BTC-USD") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        data_24h, data_30d, data_90d = self.get_historical_data(product_id)

        return tuple(self.add_technical_indicators(df) if df is not None else None for df in [data_24h, data_30d, data_90d])

    def __del__(self):
        """Cleanup method to properly close connections."""
        if self._cached_conn:
            self._cached_conn.close()


def main():
    """Main function to demonstrate the usage of Autotrade."""
    try:
        client = Autotrade()

        # Get and log balances
        usd_balance, btc_balance = client.get_available_balance()
        logger.info(f"USD balance: {usd_balance:.5f}")
        logger.info(f"BTC balance: {btc_balance:.5f}")

        # Get and analyze market data
        analysis_24h, analysis_30d, analysis_90d = client.get_analysis("BTC-USD")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == '__main__':
    main()