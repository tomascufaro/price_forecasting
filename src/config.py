
import os
from dotenv import load_dotenv, find_dotenv
from dataclasses import dataclass

#load_dotenv(find_dotenv())
load_dotenv('src/.env')
@dataclass(frozen=True)

class APIkeys:
    alpaca_api_key: str = os.getenv('alpaca_api_key')
    alpaca_secret_key: str = os.getenv('alpaca_secret_key')