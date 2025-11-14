'''
Student: Carlos Sanchez
Teacher: Steve Geinitz
Class: Machine Learning - CS-3120-001
'''

from polygon import RESTClient
import pandas as pd
import time
import csv
from stock_class import StockData
api_k = "uMcnW2vhdDQy9S956cDTSUjjoUNe1FQg"

# client = RESTClient(api_key=api_k)

# def make_df(aggs):
#     price_data = pd.DataFrame(aggs)
#     price_data['Date'] = pd.to_datetime(price_data['timestamp']*1000000).dt.date
#     price_data = price_data.set_index('Date')
#     price_data.drop(columns=['timestamp'], inplace=True)
#     return price_data

# ticker = 'SPY'
# stock_historical = client.get_aggs(ticker=ticker,
#                                multiplier=1,
#                                timespan='day',
#                                from_='2025-01-01',
#                                to='2026-01-01')

# df_historical = make_df(stock_historical)
# df_historical.to_csv("spy_historical_v2.csv")

#instantiate object class
stock = StockData(api_k)

#getting historical data of SPY500 starting date to end-date
spy_df = stock.get_spy_historical("2025-01-01", "2025-11-07")
spy_df_indicator = stock.add_indicators(spy_df)

stock.visualize(spy_df)
#spy_df.to_csv("spy_h.csv")
#spy_df_indicator.to_csv("spy_hist_indicator.csv")


# #applying indicators such as EMA, MACD ect 
# spy_df = stock.add_indicators(spy_df)

# #saving the training data
# #spy_df.to_csv("spy_training_dataset_example.csv")
# print("💾 Saved training datasets successfully!")

# # Optional visualization
# stock.visualize(spy_df)