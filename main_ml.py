from polygon import RESTClient
import pandas as pd
api_k = "uMcnW2vhdDQy9S956cDTSUjjoUNe1FQg"

client = RESTClient(api_key=api_k)

aggs = client.get_aggs("SPY", 1, "day", "2025-10-01", "2025-10-31")
price_data = pd.DataFrame(aggs)

#converting timestamp to pd datetime
price_data['Date'] = pd.to_datetime(price_data['timestamp'] * 1000000).dt.date
price_data = price_data.set_index('Date')
print(price_data)