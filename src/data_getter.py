from typing import Optional
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import fire
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.path import DATA_DIR
from src.logger import get_console_logger
import src.config as cfg

logger = get_console_logger(name='dataset_generation')

# Initialize Alpaca client
client = StockHistoricalDataClient(cfg.APIkeys.alpaca_api_key, cfg.APIkeys.alpaca_secret_key)

def download_ohlc_data_from_alpaca(
    symbol: Optional[str] = "AAPL",
    from_day: Optional[str] = "2023-01-01",
    to_day: Optional[str] = "2023-02-01",
    rewrite:Optional[bool] = False,
) -> Path:
    """
    Downloads historical OHLC data from Alpaca API and saves data to disk
    """
    # create list of days as strings
    days = pd.date_range(start=from_day, end=to_day, freq="1D")
    # create empty dataframe
    data = pd.DataFrame()
    # create download dir folder if it doesn't exist
    download_dir = DATA_DIR / 'downloads'
    download_dir.mkdir(parents=True, exist_ok=True)
    for day in days:
        day_str = day.strftime("%Y-%m-%d")
        file_name = download_dir / f'{day_str}.parquet'
        if file_name.exists():
            if rewrite:
                data_one_day = download_data_for_one_day(symbol, day_str)
                if len(data_one_day) > 0:
                    logger.info(f'Downloading data for {day_str}')
                    data_one_day.to_parquet(file_name, index=False)
                else:
                    continue
            else:                   
                logger.info(f'File {file_name} already exists, skipping')
                data_one_day = pd.read_parquet(file_name)
        else:
            logger.info(f'Downloading data for {day_str}')
            data_one_day = download_data_for_one_day(symbol, day_str)
            if len(data_one_day) > 0:
                data_one_day.to_parquet(file_name, index=False)
            else:
                continue
        # combine today's file with the rest of the data
        data = pd.concat([data, data_one_day], ignore_index=True)
    # save data to disk   
    output_file = DATA_DIR / f"ohlc_data.parquet"
    data.to_parquet(output_file, index=False)
    return output_file

def download_data_for_one_day(symbol: str, day: str) -> pd.DataFrame:
    """
    Downloads one day of data and returns a pandas DataFrame
    """
    # Convert day string to datetime object
    start = datetime.strptime(day, "%Y-%m-%d")
    end = start + timedelta(days=1)

    # Request data from Alpaca
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Hour,
        start=start,
        end=end
    )
    bars = client.get_stock_bars(request_params)

    # Transform bars to pandas DataFrame
    data = bars.df.reset_index()
    return data#[['timestamp', 'low', 'high', 'open', 'close', 'trade_count']].rename(columns={'timestamp':'time', 'trade_count':'volume'})

if __name__== '__main__':
    fire.Fire(download_ohlc_data_from_alpaca)
