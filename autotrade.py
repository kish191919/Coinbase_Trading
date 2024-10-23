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
import requests
import uuid
import robin_stocks as r
from json import dumps
import yfinance as yf
import fear_and_greed
from openai import OpenAI
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi

from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.utils import dropna
from typing import Tuple
from coinbase.rest import RESTClient

import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any

from typing import List, Dict, Tuple, Optional, Any
import traceback

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


# Configuration class
class Config:
    ROBINHOOD_USERNAME = os.getenv("username")
    ROBINHOOD_PASSWORD = os.getenv("password")
    ROBINHOOD_TOTP_CODE = os.getenv("totpcode")
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    ALPHA_VANTAGE_API_KEY = os.getenv("Alpha_Vantage_API_KEY")
    SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
    SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class TradingDecision(BaseModel):
    decision: str
    percentage: int
    reason: str

class FeeCalculator:
    """Calculate Coinbase fees based on transaction amount and payment method"""

    MINIMUM_ORDER_SIZE = 1.00  # Minimum order size in USD

    @staticmethod
    def calculate_flat_fee(amount: float) -> float:
        """Calculate flat fee based on transaction amount"""
        if amount < FeeCalculator.MINIMUM_ORDER_SIZE:
            return 0  # Don't calculate fees for orders below minimum
        elif amount <= 10:
            return 0.99
        elif amount <= 25:
            return 1.49
        elif amount <= 50:
            return 1.99
        elif amount <= 200:
            return 2.99
        return 0  # For amounts > $200, percentage-based fee applies instead

    @staticmethod
    def calculate_fees(amount: float, payment_method: str = "USD_WALLET") -> Tuple[float, float]:
        """
        Calculate total fees for a transaction
        Returns (spread_fee, transaction_fee)
        """
        # Don't calculate fees for orders below minimum
        if amount < FeeCalculator.MINIMUM_ORDER_SIZE:
            return 0.0, 0.0

        # Round amount to 2 decimal places
        amount = round(amount, 2)

        # Spread fee (0.50%)
        spread_fee = round(amount * 0.005, 2)

        # Transaction fee based on payment method
        if amount > 200:
            # Percentage-based fee for amounts over $200
            if payment_method == "ACH":
                transaction_fee = 0
            elif payment_method == "USD_WALLET":
                transaction_fee = round(amount * 0.0149, 2)  # 1.49%
            elif payment_method == "CARD":
                transaction_fee = round(amount * 0.0399, 2)  # 3.99%
            elif payment_method == "WIRE":
                transaction_fee = 10  # Incoming wire fee ($10)
            else:
                transaction_fee = round(amount * 0.0149, 2)  # Default to USD Wallet fee
        else:
            # Flat fee for amounts <= $200
            transaction_fee = FeeCalculator.calculate_flat_fee(amount)

        return spread_fee, transaction_fee


class DatabaseManager:
    def __init__(self):
        """Initialize DatabaseManager with logging and database connection"""
        self.db_name = "bitcoin_trades.db"
        self.conn = None
        self.cursor = None

        # Initialize logger first
        self.logger = logging.getLogger("DatabaseManager")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Setup database after logger is initialized
        self.setup_database()

    def setup_database(self):
        """Setup database connection and create tables if they don't exist"""
        try:
            self.conn = sqlite3.connect(self.db_name)
            self.cursor = self.conn.cursor()

            # Create trades table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    decision TEXT NOT NULL,
                    percentage INTEGER NOT NULL,
                    reason TEXT,
                    btc_balance REAL,
                    usd_balance REAL,
                    current_btc_price REAL,
                    average_buy_price REAL,
                    profit_loss REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()

            self.logger.info(f"Database setup complete: {self.db_name}")

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Database setup error: {str(e)}")
                self.logger.error(traceback.format_exc())
            else:
                print(f"Database setup error: {str(e)}")
                print(traceback.format_exc())
            raise

    def save_trade(self, trade_data: dict):
        """Save trade data to database"""
        try:
            # Ensure all required fields are present with default values
            trade_data = {
                'timestamp': trade_data.get('timestamp', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')),
                'decision': trade_data.get('decision', 'UNKNOWN'),
                'percentage': trade_data.get('percentage', 0),
                'reason': trade_data.get('reason', ''),
                'btc_balance': trade_data.get('btc_balance', 0.0),
                'usd_balance': trade_data.get('usd_balance', 0.0),
                'current_btc_price': trade_data.get('current_btc_price', 0.0),
                'average_buy_price': trade_data.get('average_buy_price', 0.0),
                'profit_loss': trade_data.get('profit_loss', 0.0)
            }

            query = '''
                INSERT INTO trades (
                    timestamp, decision, percentage, reason,
                    btc_balance, usd_balance, current_btc_price,
                    average_buy_price, profit_loss
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''

            self.cursor.execute(query, (
                trade_data['timestamp'],
                trade_data['decision'],
                trade_data['percentage'],
                trade_data['reason'],
                trade_data['btc_balance'],
                trade_data['usd_balance'],
                trade_data['current_btc_price'],
                trade_data['average_buy_price'],
                trade_data['profit_loss']
            ))
            self.conn.commit()
            self.logger.info("Trade data saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving trade data: {str(e)}")
            self.logger.error(traceback.format_exc())
            if self.conn:
                self.conn.rollback()
            raise

    def get_trades(self, limit: int = None, start_date: str = None, end_date: str = None):
        """
        Retrieve trade records with optional filtering
        """
        try:
            query = "SELECT * FROM trades"
            params = []

            # Add date filters if provided
            if start_date or end_date:
                conditions = []
                if start_date:
                    conditions.append("timestamp >= ?")
                    params.append(f"{start_date} 00:00:00")
                if end_date:
                    conditions.append("timestamp <= ?")
                    params.append(f"{end_date} 23:59:59")
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

            # Add ordering
            query += " ORDER BY timestamp DESC"

            # Add limit if provided
            if limit:
                query += f" LIMIT {limit}"

            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)

            columns = [description[0] for description in self.cursor.description]
            trades = []

            for row in self.cursor.fetchall():
                trade_dict = dict(zip(columns, row))
                trades.append(trade_dict)

            return trades

        except Exception as e:
            self.logger.error(f"Error fetching trades: {str(e)}")
            self.logger.error(traceback.format_exc())
            return []

    def __del__(self):
        """Clean up database connection on object destruction"""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
                if hasattr(self, 'logger'):
                    self.logger.info("Database connection closed")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error closing database connection: {str(e)}")
            else:
                print(f"Error closing database connection: {str(e)}")


class CoinbaseAutoTrading:
    """
    A class for automated cryptocurrency trading using Coinbase API.
    Handles authentication, data retrieval, and technical analysis.
    """

    def __init__(self):
        # Initialize CoinbaseAutoTrading with API credentials and connection settings
        self.api_key, self.api_secret, self.rest_client = self._get_api_credentials()
        self.base_url = "api.coinbase.com"
        self.exchange_url = "api.exchange.coinbase.com"
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.logger = logging.getLogger(f"CoinbaseAutoTrading")
        self.current_btc_price = 0

        self.db_manager = DatabaseManager()
        self.logger = logging.getLogger(f"CoinbaseAutoTrading")

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

        rest_client = RESTClient(api_key=api_key, api_secret=api_secret)
        if not api_key or not api_secret:
            raise ValueError("API credentials not found in environment variables")
        return api_key, api_secret, rest_client

    # Build JWT token for API authentication.
    def _build_jwt(self, method: str, resource: str) -> str:
        """
        Build JWT token for API authentication with correct method and resource path

        Args:
            method: HTTP method (GET, POST, etc.)
            resource: API resource path
        """
        private_key_bytes = self.api_secret.encode('utf-8')
        private_key = serialization.load_pem_private_key(private_key_bytes, password=None)

        # Construct the full URI path correctly
        uri = f"{method} api.coinbase.com{resource}"

        jwt_payload = {
            'sub': self.api_key,
            'iss': "coinbase-cloud",
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
    def _get_connection(self, method: str, resource: str, base_url: str) -> Tuple[dict, http.client.HTTPSConnection]:
        current_time = time.time()

        # Check if we need to refresh headers
        if (not self._cached_headers or
                not self._cached_conn or
                current_time - self._headers_timestamp > self.HEADERS_EXPIRY):

            if self._cached_conn:
                self._cached_conn.close()

            jwt_token = self._build_jwt(method, resource)
            self._cached_conn = http.client.HTTPSConnection(base_url)
            self._cached_headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {jwt_token}',
                'User-Agent': 'CoinbaseAutoTrading/1.0'
            }
            self._headers_timestamp = current_time

        return self._cached_headers, self._cached_conn

    # Get available balances for USD and BTC.
    @lru_cache(maxsize=32)
    def get_available_balance(self) -> Tuple[float, float]:
        """Get available balances for USD and BTC."""
        resource = "/api/v3/brokerage/accounts"
        headers, conn = self._get_connection("GET", resource, self.base_url)

        try:
            conn.request("GET", resource, '', headers)
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

    # Validate API response status code.
    @staticmethod
    def _check_response(response: http.client.HTTPResponse) -> None:
        if response.status != 200:
            error_msg = f"API request failed with status code: {response.status}"
            logger.error(error_msg)
            raise ValueError(error_msg)

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
                'User-Agent': 'CoinbaseAutoTrading/1.0'
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
        for window in [10, 30]:
            df[f'sma_{window}'] = SMAIndicator(close=df['close'], window=window).sma_indicator()

        return df

    # Get historical price data for different time periods.
    def get_historical_data(self, product_id: str = "BTC-USD") -> Tuple[
        Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        end_time = datetime.utcnow()
        periods = {
            '24h': {'days': 1, 'granularity': 900},  # 15-minute intervals
            '30d': {'days': 30, 'granularity': 86400}  # 1-day intervals
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

        self.current_btc_price = results[1]["close"][-1]

        return tuple(results)

    # Get technical analysis for different time periods.
    def get_analysis(self, product_id: str = "BTC-USD") -> Tuple[pd.DataFrame, pd.DataFrame]:

        data_24h, data_30d = self.get_historical_data(product_id)

        return tuple(self.add_technical_indicators(df) if df is not None else None for df in [data_24h, data_30d])

    # Cleanup method to properly close connections
    def __del__(self):
        if self._cached_conn:
            self._cached_conn.close()

    # Data Collection - VIX Index
    def get_vix_index(self):
        # Fetch VIX INDEX data
        self.logger.info("Fetching VIX INDEX data")
        try:
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="1d")
            current_vix = round(vix_data['Close'].iloc[-1], 2)
            self.logger.info(f"Current VIX INDEX: {current_vix}")
            return current_vix
        except Exception as e:
            self.logger.error(f"Error fetching VIX INDEX: {str(e)}")
            return None

    # Data Collection - Fear & Greed Index
    def get_fear_and_greed_index(self):
        # Fetch Fear and Greed Index
        self.logger.info("Fetching Fear and Greed Index")
        fgi = fear_and_greed.get()
        return {
            "value": fgi.value,
            "description": fgi.description,
            "last_update": fgi.last_update.isoformat()
        }

    # Data Collection - News
    def get_news(self):
        # Fetch news from multiple sources
        return {
            "alpha_vantage_news": self._get_news_from_alpha_vantage(),
            "robinhood_news": self._get_news_from_robinhood()
        }

    def _get_news_from_robinhood(self):
        self.logger.info("Fetching news from Robinhood")
        try:
            news_data = r.robinhood.stocks.get_news("BTC")
            news_items = []
            for item in news_data[:5]:  # Limit to 5 news items
                news_items.append({
                    'title': item['title'],
                    'published_at': item['published_at']
                })
            self.logger.info(f"Retrieved {len(news_items)} news items from Robinhood")
            return news_items
        except Exception as e:
            self.logger.error(f"Error fetching news from Robinhood: {str(e)}")
            return []

    def _get_news_from_alpha_vantage(self):
        # Fetch news from Alpha Vantage
        self.logger.info("Fetching news from Alpha Vantage")
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=BTC&apikey={Config.ALPHA_VANTAGE_API_KEY}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if "feed" not in data:
                self.logger.warning("No news data found in Alpha Vantage response")
                return []
            news_items = []
            for item in data["feed"][:10]:
                title = item.get("title", "No title")
                time_published = item.get("time_published", "No date")
                if time_published != "No date":
                    dt = datetime.strptime(time_published, "%Y%m%dT%H%M%S")
                    time_published = dt.strftime("%Y-%m-%d %H:%M:%S")
                news_items.append({
                    'title': title,
                    'pubDate': time_published
                })
            self.logger.info(f"Retrieved {len(news_items)} news items from Alpha Vantage")
            return news_items
        except Exception as e:
            self.logger.error(f"Error during Alpha Vantage API request: {e}")
            return []

    # Data Collection - Youtube
    def get_youtube_transcript(self):
        # Fetch YouTube video transcript
        video_id = 'uagC-2UjAO0'
        self.logger.info(f"Fetching YouTube transcript for video ID: {video_id}")
        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
            full_transcript = " ".join(item['text'] for item in transcript_data)
            self.logger.info(f"Retrieved transcript with {len(full_transcript)} characters")
            return full_transcript.strip()
        except Exception as e:
            self.logger.error(f"Error fetching YouTube transcript: {str(e)}")
            return f"An error occurred: {str(e)}"

    # Execute trade based on AI decision
    def execute_trade(self, decision: TradingDecision) -> bool:
        try:
            if decision.decision.upper() == "BUY":
                return self._execute_buy_order(decision.percentage)
            elif decision.decision.upper() == "SELL":
                return self._execute_sell_order(decision.percentage)
            return True  # HOLD case
        except Exception as e:
            self.logger.error(f"Trade execution failed: {str(e)}")
            return False

    # Execute buy order with Coinbase fee structure
    def _execute_buy_order(self, percentage: int) -> bool:
        """Execute buy order with Coinbase fee structure and proper USD precision"""
        try:
            # Get current balances
            usd_balance, _ = self.get_available_balance()

            # Calculate order amount based on percentage
            base_amount = usd_balance * (percentage / 100)

            # Calculate fees
            spread_fee, transaction_fee = FeeCalculator.calculate_fees(base_amount, "USD_WALLET")
            total_fees = spread_fee + transaction_fee

            # Calculate final amount after fees
            available_funds = base_amount - total_fees

            # Round to 2 decimal places for USD
            available_funds = round(available_funds, 2)

            # Minimum order size check
            if available_funds < FeeCalculator.MINIMUM_ORDER_SIZE:
                self.logger.warning(
                    f"Order amount ${available_funds:.2f} is below minimum order size ${FeeCalculator.MINIMUM_ORDER_SIZE}")
                return False

            if available_funds <= 0:
                self.logger.warning(f"Insufficient USD balance after fees. "
                                    f"Balance: ${base_amount:.2f}, "
                                    f"Fees: ${total_fees:.2f} "
                                    f"(Spread: ${spread_fee:.2f}, Transaction: ${transaction_fee:.2f})")
                return False

            # Pre-order logging
            self.logger.info(f"Attempting to place buy order:")
            self.logger.info(f"Original amount: ${base_amount:.2f}")
            self.logger.info(f"Fees: ${total_fees:.2f}")
            self.logger.info(f"Final order amount: ${available_funds:.2f}")

            # Create market buy order
            client_order_id = str(uuid.uuid4().hex)
            product_id = "BTC-USD"

            # Convert to string with proper precision
            quote_size = f"{available_funds:.2f}"

            self.logger.info(f"Placing order with quote_size: {quote_size}")

            order_data = self.rest_client.market_order_buy(
                client_order_id=client_order_id,
                product_id=product_id,
                quote_size=quote_size
            )

            order_data = dumps(order_data, indent=2)
            self.logger.info(f"Buy order executed successfully: {order_data}")
            self.logger.info(f"Order amount: ${available_funds:.2f}")
            self.logger.info(f"Total fees: ${total_fees:.2f} "
                             f"(Spread: ${spread_fee:.2f}, Transaction: ${transaction_fee:.2f})")

            return True

        except Exception as e:
            self.logger.error(f"Error executing buy order: {str(e)}")
            return False

    # Execute sell order with Coinbase fee structure
    def _execute_sell_order(self, percentage: int) -> bool:
        """Execute sell order with Coinbase fee structure and proper BTC precision"""
        try:
            # Get current balances
            _, btc_balance = self.get_available_balance()

            # Calculate amount to sell based on percentage
            btc_amount = btc_balance * (percentage / 100)

            if btc_amount <= 0:
                self.logger.warning("Insufficient BTC balance for sell order")
                return False

            # Round BTC amount to 8 decimal places (Coinbase standard)
            btc_amount = round(btc_amount, 8)

            if btc_amount < 0.00001:  # Minimum trade amount check
                self.logger.warning(f"BTC amount {btc_amount} is below minimum tradeable amount")
                return False

            # Get current BTC price using recent candle data
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=5)
            start_str = start_time.isoformat().split('+')[0] + 'Z'
            end_str = end_time.isoformat().split('+')[0] + 'Z'

            price_data = self.get_candles(
                product_id="BTC-USD",
                granularity=60,
                start=quote(start_str),
                end=quote(end_str)
            )

            if price_data is None or price_data.empty:
                self.logger.error("Unable to get current BTC price")
                return False

            current_price = float(price_data['close'].iloc[-1])
            estimated_usd_value = btc_amount * current_price

            # Calculate fees
            spread_fee, transaction_fee = FeeCalculator.calculate_fees(estimated_usd_value, "USD_WALLET")
            total_fees = spread_fee + transaction_fee

            # Create market sell order with correct authentication
            client_order_id = str(uuid.uuid4().hex)
            product_id = "BTC-USD"

            # Convert to string with proper precision
            base_size = f"{btc_amount:.8f}"

            # Log pre-order details
            self.logger.info(f"Attempting to sell {base_size} BTC")
            self.logger.info(f"Estimated USD value: ${estimated_usd_value:.2f}")

            order_data = self.rest_client.market_order_sell(
                client_order_id=client_order_id,
                product_id=product_id,
                base_size=base_size
            )

            order_data = dumps(order_data, indent=2)
            self.logger.info(f"Sell order executed successfully: {order_data}")
            self.logger.info(f"Amount sold: {base_size} BTC")
            self.logger.info(f"Estimated total fees: ${total_fees:.2f} "
                             f"(Spread: ${spread_fee:.2f}, Transaction: ${transaction_fee:.2f})")

            return True

        except Exception as e:
            self.logger.error(f"Error executing sell order: {str(e)}")
            return False


    # Perform AI-based analysis
    def ai_analysis(self):
        # Get and analyze market data
        analysis_24h, analysis_30d = self.get_analysis("BTC-USD")
        news = self.get_news()
        fgi = self.get_fear_and_greed_index()
        vix_index = self.get_vix_index()
        # youtube_transcript = self.get_youtube_transcript()
        # f = open("strategy.txt","w",encoding="utf-8")
        # f.write(youtube_transcript)
        f = open("strategy.txt", "r")
        youtube_transcript = f.read()
        f.close()

        self.logger.info("Sending request to OpenAI")
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an expert in BTC investing. Analyze the provided data and determine whether to buy, sell, or hold at the current moment. Consider the following in your analysis:
                        - Technical indicators and market data
                        - Recent news headlines and their potential impact on BTC price
                        - The Fear and Greed Index and its implications
                        - VIX INDEX and its implications for market volatility
                        - Current VIX INDEX: {vix_index}
                        
                        Particularly important is to always refer to the trading method of 'TheMovingAverage', to assess the current situation and make trading decisions. TheMovingAverage's trading method is as follows:

                        {youtube_transcript}

                        Based on this trading method, analyze the current market situation and make a judgment by synthesizing it with the provided data.

                        Response format:
                        1. Decision (buy, sell, or hold)
                        2. If the decision is 'buy', provide a percentage (1-100) of available KRW to use for buying.
                        If the decision is 'sell', provide a percentage (1-100) of held BTC to sell.
                        If the decision is 'hold', set the percentage to 0.
                        3. Reason for your decision

                        Ensure that the percentage is an integer between 1 and 100 for buy/sell decisions, and exactly 0 for hold decisions.
                        Your percentage should reflect the strength of your conviction in the decision based on the analyzed data."""},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "month_data": analysis_30d.to_json(),
                                "daily_data": analysis_24h.to_json(),
                                "fear_and_greed_index": fgi,
                                "vix_index": vix_index,
                                "news": news
                            })
                        }
                    ]
                }
            ],
            max_tokens=4095,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "trading_decision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "decision": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                            "percentage": {"type": "integer"},
                            "reason": {"type": "string"}
                        },
                        "required": ["decision", "percentage", "reason"],
                        "additionalProperties": False
                    }
                }
            }
        )
        result = TradingDecision.model_validate_json(response.choices[0].message.content)
        self.logger.info("Received response from OpenAI")
        self.logger.info(f"### AI Decision: {result.decision.upper()} ###")
        self.logger.info(f"### Percentage: {result.percentage} ###")
        self.logger.info(f"### Reason: {result.reason} ###")
        self.logger.info(f"### VIX INDEX: {vix_index} ###")

        # Execute trade
        self.execute_trade(result)

        # Get trading data
        trading_stats = self.get_trading_data()

        # Calculate profit if trading_stats is available
        if trading_stats is not None:
            self.calculate_trading_profit(trading_stats)

        # Save to database
        self.save_trading_decision(result, trading_stats)

    def save_trading_decision(self, decision: TradingDecision, trading_stats: dict):
        """거래 결정 및 결과를 데이터베이스에 저장"""
        try:
            # 현재 잔고 조회
            usd_balance, btc_balance = self.get_available_balance()

            # 저장할 데이터 구성
            trade_data = {
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                'decision': decision.decision,
                'percentage': decision.percentage,
                'reason': decision.reason,
                'btc_balance': btc_balance,
                'usd_balance': usd_balance,
                'current_btc_price': self.current_btc_price,
                'average_buy_price': trading_stats.get('average_buy_price', 0),
                'profit_loss': trading_stats.get('profit_loss', 0)
            }

            # 데이터베이스에 저장
            self.db_manager.save_trade(trade_data)
            self.logger.info("Trade data saved to database successfully")

        except Exception as e:
            self.logger.error(f"Error saving trade data to database: {str(e)}")
            raise

    def get_trading_data(self):
        """
        Calculate BTC trading statistics from historical fills.
        """
        try:
            # Get fills using the REST client
            fills_response = self.rest_client.get_fills(
                product_id='BTC-USD'
            )

            # Initialize default trading stats
            default_trading_stats = {
                'net_btc_position': 0.0,
                'average_buy_price': 0.0,
                'average_sell_price': 0.0,
                'total_commission': 0.0,
                'cost_basis': 0.0,
                'current_value': 0.0,
                'roi_percentage': 0.0,
                'current_price': self.current_btc_price or 0.0,
                'trade_summary': {
                    'total_buy_btc': 0.0,
                    'total_buy_cost': 0.0,
                    'total_sell_btc': 0.0,
                    'total_sell_proceeds': 0.0,
                    'completed_buys': [],
                    'completed_sells': []
                },
                'profit_loss': 0.0
            }

            # Debug logging and response handling
            self.logger.debug(f"Raw fills response type: {type(fills_response)}")
            if isinstance(fills_response, dict):
                fills = fills_response.get('fills', [])
                self.logger.debug(f"Found {len(fills)} fills in response")
            elif isinstance(fills_response, str):
                try:
                    response_dict = json.loads(fills_response)
                    fills = response_dict.get('fills', [])
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse JSON: {e}")
                    return default_trading_stats
            else:
                self.logger.error(f"Unexpected response type: {type(fills_response)}")
                return default_trading_stats

            if not fills:
                self.logger.warning("No fills data available")
                return default_trading_stats

            # Initialize tracking variables
            total_buy_btc = 0.0
            total_buy_cost = 0.0
            total_sell_btc = 0.0
            total_sell_proceeds = 0.0
            total_commission = 0.0
            completed_buys = []  # Track completed buy trades
            completed_sells = []  # Track completed sell trades

            for fill in fills:
                try:
                    price = float(fill.get('price', 0))
                    commission = float(fill.get('commission', 0))
                    is_buy = fill.get('side') == 'BUY'
                    size_in_quote = fill.get('size_in_quote', False)
                    size = float(fill.get('size', 0))

                    # Convert size to BTC amount if size_in_quote is True
                    btc_amount = size / price if size_in_quote else size
                    usd_amount = size if size_in_quote else size * price

                    # Track commission
                    total_commission += commission

                    if is_buy:
                        total_buy_btc += btc_amount
                        total_buy_cost += usd_amount
                        # Store completed buy trades
                        completed_buys.append({
                            'btc_amount': btc_amount,
                            'usd_amount': usd_amount,
                            'price': price,
                            'timestamp': fill.get('timestamp', 'N/A')
                        })
                    else:
                        total_sell_btc += btc_amount
                        total_sell_proceeds += usd_amount
                        # Store completed sell trades
                        completed_sells.append({
                            'btc_amount': btc_amount,
                            'usd_amount': usd_amount,
                            'price': price,
                            'timestamp': fill.get('timestamp', 'N/A')
                        })
                except (ValueError, TypeError) as e:
                    self.logger.error(f"Error processing fill: {str(e)}")
                    continue

            # Calculate average buy price from completed buys
            if completed_buys:
                weighted_buy_sum = sum(buy['usd_amount'] for buy in completed_buys)
                total_bought_btc = sum(buy['btc_amount'] for buy in completed_buys)
                average_buy_price = weighted_buy_sum / total_bought_btc if total_bought_btc > 0 else 0
            else:
                average_buy_price = 0

            # Calculate average sell price from completed sells
            if completed_sells:
                weighted_sell_sum = sum(sell['usd_amount'] for sell in completed_sells)
                total_sold_btc = sum(sell['btc_amount'] for sell in completed_sells)
                average_sell_price = weighted_sell_sum / total_sold_btc if total_sold_btc > 0 else 0
            else:
                average_sell_price = 0

            # Calculate net BTC position (considering commission)
            net_btc_position = total_buy_btc - total_sell_btc - total_commission

            # Calculate current value and ROI
            if self.current_btc_price and net_btc_position > 0:
                current_value = net_btc_position * self.current_btc_price
                cost_basis = total_buy_cost - total_sell_proceeds
                roi = ((current_value - cost_basis) / cost_basis * 100) if cost_basis > 0 else 0
            else:
                current_value = net_btc_position * (price if 'price' in locals() else 0)
                cost_basis = total_buy_cost - total_sell_proceeds
                roi = 0

            # Calculate profit/loss
            profit_loss = total_sell_proceeds - (total_buy_cost + total_commission)

            trading_stats = {
                'net_btc_position': net_btc_position,
                'average_buy_price': average_buy_price,
                'average_sell_price': average_sell_price,
                'total_commission': total_commission,
                'cost_basis': cost_basis,
                'current_value': current_value,
                'roi_percentage': roi,
                'current_price': self.current_btc_price,
                'profit_loss': profit_loss,
                'trade_summary': {
                    'total_buy_btc': total_buy_btc,
                    'total_buy_cost': total_buy_cost,
                    'total_sell_btc': total_sell_btc,
                    'total_sell_proceeds': total_sell_proceeds,
                    'completed_buys': completed_buys,
                    'completed_sells': completed_sells
                }
            }

            self.logger.info(f"Trading stats calculated successfully")
            self.logger.debug(f"Trading stats: {trading_stats}")
            return trading_stats

        except Exception as e:
            self.logger.error(f"Error in get_trading_data: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def calculate_trading_profit(self, trading_stats):
        """
        Calculate trading profits and returns from trading statistics including weighted average prices
        """
        try:
            if trading_stats is None:
                self.logger.warning("No trading stats available for profit calculation")
                return None

            # Extract required values with safe gets
            trade_summary = trading_stats.get('trade_summary', {})
            total_buy_cost = trade_summary.get('total_buy_cost', 0.0)
            total_sell_proceeds = trade_summary.get('total_sell_proceeds', 0.0)
            total_commission = trading_stats.get('total_commission', 0.0)
            completed_buys = trade_summary.get('completed_buys', [])
            completed_sells = trade_summary.get('completed_sells', [])

            # Calculate total costs (buy cost + commission)
            total_costs = total_buy_cost + total_commission

            # Calculate profit/loss
            profit_loss = total_sell_proceeds - total_costs

            # Calculate return percentage
            return_percentage = (profit_loss / total_costs * 100) if total_costs > 0 else 0

            # Get average prices
            average_buy_price = trading_stats.get('average_buy_price', 0.0)
            average_sell_price = trading_stats.get('average_sell_price', 0.0)

            # Calculate total traded BTC
            total_bought_btc = sum(buy.get('btc_amount', 0.0) for buy in completed_buys)
            total_sold_btc = sum(sell.get('btc_amount', 0.0) for sell in completed_sells)

            # Create results dictionary
            results = {
                'total_buy_cost': total_buy_cost,
                'total_sell_proceeds': total_sell_proceeds,
                'total_commission': total_commission,
                'total_costs': total_costs,
                'profit_loss': profit_loss,
                'return_percentage': return_percentage,
                'average_buy_price': average_buy_price,
                'average_sell_price': average_sell_price,
                'total_btc_bought': total_bought_btc,
                'total_btc_sold': total_sold_btc
            }

            # Print detailed results
            print("\n=== Trading Profit Analysis ===")
            print(f"Total Buy Cost: ${total_buy_cost:,.2f}")
            print(f"Total Sell Proceeds: ${total_sell_proceeds:,.2f}")
            print(f"Total Commission: ${total_commission:,.2f}")
            print(f"Total Costs (Buy + Commission): ${total_costs:,.2f}")
            print(f"Average Buy Price: ${average_buy_price:,.2f}")
            print(f"Average Sell Price: ${average_sell_price:,.2f}")
            print(f"Total BTC Bought: {total_bought_btc:.8f} BTC")
            print(f"Total BTC Sold: {total_sold_btc:.8f} BTC")
            print(f"Profit/Loss: ${profit_loss:,.2f}")
            print(f"Return: {return_percentage:.2f}%")

            # Print individual buy trades
            if completed_buys:
                print("\nCompleted Buy Trades:")
                for i, buy in enumerate(completed_buys, 1):
                    print(f"Buy {i}: {buy.get('btc_amount', 0):.8f} BTC @ ${buy.get('price', 0):,.2f}")

            # Print individual sell trades
            if completed_sells:
                print("\nCompleted Sell Trades:")
                for i, sell in enumerate(completed_sells, 1):
                    print(f"Sell {i}: {sell.get('btc_amount', 0):.8f} BTC @ ${sell.get('price', 0):,.2f}")

            print("===========================\n")

            return results

        except Exception as e:
            self.logger.error(f"Error calculating trading profit: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

def main():
    """Main function to demonstrate the usage of CoinbaseAutoTrading."""
    try:
        client = CoinbaseAutoTrading()
        # Get and log balances
        usd_balance, btc_balance = client.get_available_balance()
        logger.info(f"USD balance: {usd_balance:.5f}")
        logger.info(f"BTC balance: {btc_balance:.5f}")

        client.ai_analysis()
        usd_balance, btc_balance = client.get_available_balance()
        logger.info(f"USD balance: {usd_balance:.5f}")
        logger.info(f"BTC balance: {btc_balance:.5f}")


    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == '__main__':
    main()