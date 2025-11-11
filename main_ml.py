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

client = RESTClient(api_key=api_k)

def make_df(aggs):
    price_data = pd.DataFrame(aggs)
    price_data['Date'] = pd.to_datetime(price_data['timestamp']*1000000).dt.date
    price_data = price_data.set_index('Date')
    price_data.drop(columns=['timestamp'], inplace=True)
    return price_data

# # print(make_df(aggs))

# def read_csv_with_csv_module(file_path):
#     """
#     Reads a CSV file and returns a pandas DataFrame.
#     """
#     with open(file_path, mode='r', newline='', encoding='utf-8') as file:
#         csv_reader = csv.reader(file)
#         header = next(csv_reader)

#         rows = []
#         for row in csv_reader:
#             rows.append(row)

#     df = pd.DataFrame(rows, columns=header)
#     print(f"✅ Loaded {len(df)} rows from {file_path}")
#     return df

# greeks = read_csv_with_csv_module('spy_options_with_greeks.csv')
# print(greeks.head())

# ticker = "O:SPY251110C00670000"

# dailyOptions = client.get_aggs(ticker=ticker,
#                                multiplier=1,
#                                timespan='day',
#                                from_='2025-01-01',
#                                to='2026-01-01')

# op_df = make_df(dailyOptions)
# op_df.to_csv("spy_options_v2.csv", index=True)
# print(op_df)

# Example expirations (could automate this)
# expirations = [
#     "2025-01-17", "2025-02-21", "2025-03-21",
#     "2025-04-18", "2025-05-16", "2025-06-20"
# ]

# all_data = []

# for exp in expirations:
#     ticker = f"O:SPY{exp.replace('-', '')[-6:]}C00450000"  # Example strike 450 call
#     try:
#         dailyOptions = client.get_aggs(
#             ticker=ticker,
#             multiplier=1,
#             timespan='day',
#             from_='2025-01-01',
#             to='2026-01-01'
#         )
#         df = make_df(dailyOptions)
#         df['expiration_date'] = exp
#         df['symbol'] = ticker
#         all_data.append(df)
#     except Exception as e:
#         print(f"Error fetching {ticker}: {e}")

# if all_data:
#     op_df = pd.concat(all_data)
#     op_df.to_csv("spy_options_multiple.csv", index=True)
#     print(op_df)

# ticker = 'SPY'
# stock_historical = client.get_aggs(ticker=ticker,
#                                multiplier=1,
#                                timespan='day',
#                                from_='2025-01-01',
#                                to='2026-01-01')

# df_historical = make_df(stock_historical)
# df_historical.to_csv("spy_historical_v2.csv")
stock_df = StockData(api_k)

stock_spy = stock_df.get_spy_historical("2025-01-01", "2025-11-07")
stock_spy = stock_df.add_indicators(stock_spy)
# Optional visualization
stock_df.visualize(stock_spy)