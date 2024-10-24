import os
import json
import uuid
import pandas as pd
import http.client

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


def main():
    client, api_key, api_secret = get_api_key()

    get_list_of_account(api_key, api_secret)
    get_product_book(api_key, api_secret)

    bids_df, asks_df = get_order_book(product_id="BTC-USD", level=2)
    taker_fee_rate, maker_fee_rate = retrieve_fees_charged(client)
    cash_balance,btc_balance = retrieve_balance(client)

    df = get_product_candles(api_key, api_secret)
    ai_trading(df,client,btc_balance,cash_balance,taker_fee_rate,maker_fee_rate)

def get_api_key():
    load_dotenv()
    api_key = os.getenv("api_key")
    api_secret = os.getenv("api_secret")

    # Create Instance
    client = RESTClient(api_key=api_key, api_secret=api_secret)
    return client, api_key, api_secret


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


def get_list_of_account(api_key, api_secret, limit=49, cursor=None):
    resource = "accounts"
    jwt_token = build_jwt(resource, api_key, api_secret)

    # Establish a connection to view order book data from Coinbase Pro API
    conn = http.client.HTTPSConnection("api.coinbase.com")
    payload = ''
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer '+ str(jwt_token)
    }

    # Send account inquiry request
    url = f"/api/v3/brokerage/accounts"
    conn.request("GET", url, payload, headers)

    # Get response
    res = conn.getresponse()

    # Check response status code
    check_response_status_code(res)

    # Read response body
    data = res.read()

    try:
        # Parse into JSON data
        list_of_account = json.loads(data.decode("utf-8"))
        print(json.dumps(list_of_account, indent=2))
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {str(e)}")
        print(f"Raw response: {data.decode('utf-8')}")

    # Terminate connection
    conn.close()


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
            print("Error: 'bids' or 'asks' key not found in the response")
            print(f"Full response: {json.dumps(order_book, indent=2)}")
            return None, None

        # bids(매수)와 asks(매도) 추출
        bids = order_book['bids']  # [price, size, num_orders]
        asks = order_book['asks']  # [price, size, num_orders]

        # DataFrame으로 변환하여 출력
        bids_df = pd.DataFrame(bids, columns=['price', 'size', 'num_orders'])
        asks_df = pd.DataFrame(asks, columns=['price', 'size', 'num_orders'])

        print("Order Book - Bids (Buy Orders):")
        print(bids_df)

        print("\nOrder Book - Asks (Sell Orders):")
        print(asks_df)

        return bids_df, asks_df

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {str(e)}")
        print(f"Raw response: {data.decode('utf-8')}")
        return None, None


def get_product_book(api_key, api_secret):
    resource = "product_book"
    jwt_token = build_jwt(resource, api_key, api_secret)

    # Establish a connection to view products
    conn = http.client.HTTPSConnection("api.coinbase.com")
    payload = ''
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + str(jwt_token)
    }


    # Send order book inquiry request (level 2: Aggregated Order Book)
    url = f"/api/v3/brokerage/product_book?product_id=BTC-USD&limit=1"
    conn.request("GET", url, payload, headers)
    print("conn request")

    # Get response
    res = conn.getresponse()

    # Check response status code
    check_response_status_code(res)

    # Read response body
    data = res.read()

    try:
        # Parse into JSON data
        product_book = json.loads(data.decode("utf-8"))
        print(json.dumps(product_book, indent=2))
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {str(e)}")
        print(f"Raw response: {data.decode('utf-8')}")

    # Terminate connection
    conn.close()



def get_product_candles(api_key, api_secret):
    product_id = 'BTC-USD'
    granularity = "ONE_DAY"
    limit = 350
    end = datetime.now()
    start = end - timedelta(days=30)

    end_unix = int(end.timestamp())
    start_unix = int(start.timestamp())

    resource = f"products/{product_id}/candles"
    jwt_token = build_jwt(resource, api_key, api_secret)

    # Establish a connection to view product candles data from Coinbase API
    conn = http.client.HTTPSConnection("api.coinbase.com")
    payload = ''
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + str(jwt_token)
    }

    # API request with parameters
    url = f"/api/v3/brokerage/products/{product_id}/candles?start={str(start_unix)}&end={str(end_unix)}&granularity={granularity}&limit={limit}"
    conn.request("GET", url, payload, headers)

    # Get response
    res = conn.getresponse()

    # Check response status code
    check_response_status_code(res)

    # Read response body
    data = res.read()

    try:
        # Parse into JSON data
        product_candles = json.loads(data.decode("utf-8"))

        # Convert candles list to data frame
        product_candles_df = pd.DataFrame(product_candles["candles"])

        # Data type conversion (conversion to numeric type)
        product_candles_df["start"] = pd.to_numeric(product_candles_df["start"], errors='coerce')
        product_candles_df["start"] = pd.to_datetime(product_candles_df["start"], unit='s')  # timestamp 변환
        product_candles_df[["low", "high", "open", "close", "volume"]] = product_candles_df[
            ["low", "high", "open", "close", "volume"]].astype(float)

        # Dataframe output
        print(product_candles_df)
        return product_candles_df

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {str(e)}")
        print(f"Raw response: {data.decode('utf-8')}")

    # Terminate connection
    conn.close()



def retrieve_fees_charged(client):
    summary = client.get_transaction_summary()
    taker_fee_rate = summary['fee_tier']['taker_fee_rate']
    maker_fee_rate = summary['fee_tier']['maker_fee_rate']
    taker_fee_rate = float(taker_fee_rate)
    maker_fee_rate = float(maker_fee_rate)

    print("Taker Fee Rate : ", taker_fee_rate)
    print("Maker Fee Rate : ", maker_fee_rate)
    return taker_fee_rate, maker_fee_rate


def retrieve_balance(client):
    # Check Cash
    accounts = client.get_accounts()
    cash_balance = float(accounts['accounts'][0]['available_balance']['value'])

    # Check BTC
    btc_balance = float(accounts['accounts'][1]['available_balance']['value'])

    print("Cash Balance : ", cash_balance)
    print("BTC Balance : ", btc_balance)
    return cash_balance,btc_balance

def check_response_status_code(res):
    if res.status != 200:
        print(f"Error: Received status code {res.status}")
        return

def ai_trading(df,client,btc_balance,cash_balance,taker_fee_rate,maker_fee_rate):
    result = openAI_request(df)
    result['decision'] = 'hold'

    if result['decision'] == 'sell':
        client_order_id = str(uuid.uuid4().hex)
        product_id = "BTC-USD"
        base_size = trunc(btc_balance * (1 - taker_fee_rate) * 1000000) / 1000000  # BTC
        base_size = str(base_size)
        order_data = client.market_order_sell(client_order_id=client_order_id, product_id=product_id,
                                              base_size=base_size)
        print(dumps(order_data, indent=2))
        print("Sell : ", result['reason'])

    elif result['decision'] == 'buy':
        client_order_id = str(uuid.uuid4().hex)
        product_id = "BTC-USD"
        quote_size = trunc(cash_balance * (1 - maker_fee_rate) * 100) / 100  # Cash
        quote_size = str(quote_size)
        order_data = client.market_order_buy(client_order_id=client_order_id, product_id=product_id,
                                             quote_size=quote_size)
        print(dumps(order_data, indent=2))
        print("BUY : ", result['reason'])

    elif result['decision'] == 'hold':
        print("Hold : ", result['reason'])

def openAI_request(df):
    openAI_client = OpenAI()

    response = openAI_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an expert in Bitcoin investing. Tell me whether to buy, sell, or hold at the moment based on the chart data provided. response in json format.\n\nResponse Example:\n{\"decision\":\"buy\",\"reason\":\"some technical reason\"}\n{\"decision\":\"sell\",\"reason\":\"some technical reason\"}\n{\"decision\":\"hold\",\"reason\":\"some technical reason\"}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "{\"timestamp\":{\"0\":1724112000000,\"1\":1724198400000,\"2\":1724284800000,\"3\":1724371200000,\"4\":1724457600000,\"5\":1724544000000,\"6\":1724630400000,\"7\":1724716800000,\"8\":1724803200000,\"9\":1724889600000,\"10\":1724976000000,\"11\":1725062400000,\"12\":1725148800000,\"13\":1725235200000,\"14\":1725321600000,\"15\":1725408000000,\"16\":1725494400000,\"17\":1725580800000,\"18\":1725667200000,\"19\":1725753600000,\"20\":1725840000000,\"21\":1725926400000,\"22\":1726012800000,\"23\":1726099200000,\"24\":1726185600000,\"25\":1726272000000,\"26\":1726358400000,\"27\":1726444800000,\"28\":1726531200000,\"29\":1726617600000},\"open\":{\"0\":59455.81,\"1\":59017.59,\"2\":61167.54,\"3\":60383.28,\"4\":64091.63,\"5\":64179.63,\"6\":64250.01,\"7\":62840.09,\"8\":59437.68,\"9\":59047.94,\"10\":59353.12,\"11\":59117.59,\"12\":58968.37,\"13\":57291.21,\"14\":59138.89,\"15\":57460.5,\"16\":57971.0,\"17\":56156.82,\"18\":53950.0,\"19\":54159.6,\"20\":54881.1,\"21\":57047.82,\"22\":57641.15,\"23\":57352.92,\"24\":58137.33,\"25\":60543.35,\"26\":60012.34,\"27\":59122.7,\"28\":58209.76,\"29\":60317.38},\"high\":{\"0\":61457.03,\"1\":61849.98,\"2\":61430.48,\"3\":64987.0,\"4\":64529.78,\"5\":65050.08,\"6\":64509.36,\"7\":63226.26,\"8\":60236.98,\"9\":61194.09,\"10\":59929.63,\"11\":59446.99,\"12\":59070.55,\"13\":59423.0,\"14\":59825.7,\"15\":58531.25,\"16\":58326.12,\"17\":56995.0,\"18\":54847.0,\"19\":55315.95,\"20\":58119.97,\"21\":58050.35,\"22\":58014.35,\"23\":58600.0,\"24\":60670.0,\"25\":60660.0,\"26\":60402.34,\"27\":59214.15,\"28\":61373.41,\"29\":61358.0},\"low\":{\"0\":58571.96,\"1\":58793.89,\"2\":59750.0,\"3\":60343.2,\"4\":63564.22,\"5\":63793.74,\"6\":62806.8,\"7\":58025.49,\"8\":57851.62,\"9\":58729.39,\"10\":57700.0,\"11\":58739.31,\"12\":57200.0,\"13\":57119.01,\"14\":57394.49,\"15\":55555.0,\"16\":55628.04,\"17\":52530.0,\"18\":53733.1,\"19\":53623.95,\"20\":54565.56,\"21\":56377.76,\"22\":55534.41,\"23\":57311.15,\"24\":57630.01,\"25\":59436.8,\"26\":58695.75,\"27\":57477.0,\"28\":57620.27,\"29\":59174.5},\"close\":{\"0\":59017.59,\"1\":61163.28,\"2\":60383.29,\"3\":64086.72,\"4\":64179.63,\"5\":64251.93,\"6\":62840.0,\"7\":59439.64,\"8\":59045.88,\"9\":59364.47,\"10\":59112.77,\"11\":58968.37,\"12\":57299.0,\"13\":59139.83,\"14\":57468.84,\"15\":57971.0,\"16\":56156.82,\"17\":53950.01,\"18\":54156.33,\"19\":54881.11,\"20\":57053.9,\"21\":57645.59,\"22\":57352.79,\"23\":58137.54,\"24\":60543.35,\"25\":60012.35,\"26\":59122.33,\"27\":58208.75,\"28\":60312.6,\"29\":60423.1},\"volume\":{\"0\":11062.05279919,\"1\":13380.11101719,\"2\":9608.6955368,\"3\":17852.77117167,\"4\":5020.91676004,\"5\":4012.30764642,\"6\":10741.96651081,\"7\":14195.89496578,\"8\":10001.12169866,\"9\":10933.22040343,\"10\":9382.9124508,\"11\":1368.21775121,\"12\":4411.05388122,\"13\":4318.6544603,\"14\":8264.44971184,\"15\":13715.96171494,\"16\":12281.57828747,\"17\":18495.45064356,\"18\":3284.57722628,\"19\":3684.86816392,\"20\":10203.03329517,\"21\":6433.21371111,\"22\":11678.94297124,\"23\":10175.62390483,\"24\":11935.95544754,\"25\":3147.39019631,\"26\":3975.3568185,\"27\":7654.10991835,\"28\":11701.89074715,\"29\":10441.50380985}}"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": df.to_json()
                    }
                ]
            }
        ],
        response_format={
            "type": "json_object"
        }
    )

    result = json.loads(response.choices[0].message.content)
    return result


if __name__ == '__main__':
    main()







