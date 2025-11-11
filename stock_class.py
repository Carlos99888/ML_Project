'''
Student: Carlos Sanchez
Teacher: Steve Geinitz
Class: Machine Learning - CS-3120-001
'''

import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import numpy as np
import time
from datetime import datetime
import csv

class Stock_class:
    #properties
    df = None

    #constructor
    def __init__(self, dataframe):
        self.df = dataframe

    def get_spy_historical(self, start_date="2024-01-01", end_date="2024-12-31"):
        """
        Fetches daily OHLCV data for SPY between start and end dates.
        """
        data = []
        for bar in client.list_aggs(
            "SPY",                # ticker symbol
            1,                    # multiplier (1 day)
            "day",                # timespan
            start_date,
            end_date,
            adjusted=True,
            sort="asc",
            limit=50000
        ):
            # Convert integer timestamp (ms) → datetime
            date = datetime.fromtimestamp(bar.timestamp / 1000)

            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume
            })

        df = pd.DataFrame(data)
        return df