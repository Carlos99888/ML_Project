from polygon import RESTClient
import pandas as pd
import time
import csv
api_k = "uMcnW2vhdDQy9S956cDTSUjjoUNe1FQg"

client = RESTClient(api_key=api_k)

# aggs = client.get_aggs("O:SPY251110C00670000", timespan="day", multiplier=1,
#                 from_="2025-10-10", to="2025-11-08")
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

ticker = "O:SPY251110C00670000"

dailyOptions = client.get_aggs(ticker=ticker,
                               multiplier=1,
                               timespan='day',
                               from_='2025-01-01',
                               to='2026-01-01')

op_df = make_df(dailyOptions)
op_df.to_csv("spy_options_v2.csv", index=True)
print(op_df)