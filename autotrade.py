import os
import json
import uuid
import pandas as pd
import http.client
import logging

from json import dumps
from math import trunc
from dotenv import load_dotenv
from coinbase.rest import RESTClient
from datetime import datetime, timedelta
from openai import OpenAI

import jwt
from cryptography.hazmat.primitives import serialization
import time
import secrets

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Coinbase 객체 생성
    api_key, api_secret = get_api_key()

    # 1. 현재 투자 상태 조회
    USD_balance, BTC_balance = get_available_balance(api_key, api_secret)

    logger.info(f"USD_balance : {USD_balance:.5f}")
    logger.info(f"BTC_balance : {BTC_balance:.5f}")

    # 2. 오더북(호가 데이터) 조회
    bids_df, asks_df = get_order_book(product_id="BTC-USD", level=2)

def get_api_key():
    load_dotenv()
    api_key = os.getenv("api_key")
    api_secret = os.getenv("api_secret")
    return api_key, api_secret

def build_jwt(resource, api_key, api_secret):
    private_key_bytes = api_secret.encode('utf-8')
    private_key = serialization.load_pem_private_key(private_key_bytes, password=None)
    uri = f"GET api.coinbase.com/api/v3/brokerage/{resource}"
    jwt_payload = {
        'sub': api_key,
        'iss': "cdp",
        'nbf': int(time.time()),
        'exp': int(time.time()) + 120,
        'uri': uri,
    }
    jwt_token = jwt.encode(
        jwt_payload,
        private_key,
        algorithm='ES256',
        headers={'kid': api_key, 'nonce': secrets.token_hex()},
    )
    return jwt_token

def check_response_status_code(res):
    if res.status != 200:
        logger.error(f"Error: Received status code {res.status}")
        return

def _connect_coinbase_API(resource, api_key, api_secret, HTTPS):
    jwt_token = build_jwt(resource, api_key, api_secret)


    # Coinbase Pro API 연결
    conn = http.client.HTTPSConnection(HTTPS)
    payload = ''
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + str(jwt_token)
    }
    return headers, conn, payload

# 1. 현재 투자 상태 조회
def get_available_balance(api_key, api_secret):
    resource = "accounts"
    HTTPS = "api.coinbase.com"
    headers, conn, payload = _connect_coinbase_API(resource, api_key, api_secret, HTTPS)

    # 계정 조회 요청
    url = f"/api/v3/brokerage/accounts"
    conn.request("GET", url, payload, headers)

    # 응답 받기
    res = conn.getresponse()

    # 응답 상태 확인
    check_response_status_code(res)

    # 응답 데이터 읽기
    data = res.read()

    try:
        # JSON 데이터 파싱
        accounts = json.loads(data.decode("utf-8"))

        # 각 계정의 잔액 정보 출력
        logger.info("=== Available Balances ===")
        for account in accounts.get('accounts', []):
            currency = account.get('currency', '')
            available = float(account.get('available_balance', {}).get('value', 0))

            if available > 0:  # 잔액이 있는 계정만 출력
                if currency == 'USD':
                    USD_balance = available
                elif currency == 'BTC':
                    BTC_balance = available

    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error: {str(e)}")
        logger.error(f"Raw response: {data.decode('utf-8')}")

    # 연결 종료
    conn.close()

    return USD_balance, BTC_balance

# 2. 오더북(호가 데이터) 조회
def get_order_book(product_id="BTC-USD", level=2):
    # Establish a connection to view order book data from Coinbase Pro API
    conn = http.client.HTTPSConnection("api.exchange.coinbase.com")
    payload = ''
    headers = {
        'Content-Type': 'application/json'
    }

    # Send order book inquiry request (level 2: Aggregated Order Book)
    url = f"/products/{product_id}/book?level={level}"
    conn.request("GET", url, payload, headers)

    # Get response
    res = conn.getresponse()

    # Check response status code
    check_response_status_code(res)

    # Read response body
    data = res.read()

    try:
        # Parse into JSON data
        order_book = json.loads(data.decode("utf-8"))

        # Check if 'bids' and 'asks' keys exist in the response
        if 'bids' not in order_book or 'asks' not in order_book:
            logger.error("Error: 'bids' or 'asks' key not found in the response")
            logger.error(f"Full response: {json.dumps(order_book, indent=2)}")
            return None, None

        # bids(매수)와 asks(매도) 추출
        bids = order_book['bids']  # [price, size, num_orders]
        asks = order_book['asks']  # [price, size, num_orders]

        # DataFrame으로 변환하여 출력
        bids_df = pd.DataFrame(bids, columns=['price', 'size', 'num_orders'])
        asks_df = pd.DataFrame(asks, columns=['price', 'size', 'num_orders'])

        return bids_df, asks_df

    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error: {str(e)}")
        logger.error(f"Raw response: {data.decode('utf-8')}")
        return None, None

if __name__ == '__main__':
    main()