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


class Autotrade:
    """
    A class for automated cryptocurrency trading using Coinbase API.
    Handles authentication, data retrieval, and technical analysis.
    """

    def __init__(self):
        # Initialize Autotrade with API credentials and connection settings
        self.api_key, self.api_secret, self.rest_client = self._get_api_credentials()
        self.base_url = "api.coinbase.com"
        self.exchange_url = "api.exchange.coinbase.com"
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.logger = logging.getLogger(f"Autotrade")

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
                'User-Agent': 'AutoTrade/1.0'
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
        Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
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

        return tuple(results)

    # Get technical analysis for different time periods.
    def get_analysis(self, product_id: str = "BTC-USD") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

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

        self.execute_trade(result)


def main():
    """Main function to demonstrate the usage of Autotrade."""
    try:
        client = Autotrade()

        # Get and log balances
        usd_balance, btc_balance = client.get_available_balance()
        logger.info(f"USD balance: {usd_balance:.5f}")
        logger.info(f"BTC balance: {btc_balance:.5f}")
        client.ai_analysis()



    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == '__main__':
    main()