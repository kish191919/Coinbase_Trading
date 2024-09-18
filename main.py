from coinbase import jwt_generator
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("api_key")
api_secret = os.getenv("api_secret")

request_method = "GET"
request_path = "/v2/accounts"

def main():
    jwt_uri = jwt_generator.format_jwt_uri(request_method, request_path)
    jwt_token = jwt_generator.build_rest_jwt(jwt_uri, api_key, api_secret)
    print(f"export JWT={jwt_token}")

if __name__ == "__main__":
    main()
