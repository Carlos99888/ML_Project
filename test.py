from polygon import RESTClient
import pandas as pd
import numpy as np
import time
from datetime import datetime
import csv
from scipy.stats import norm

api_k = "uMcnW2vhdDQy9S956cDTSUjjoUNe1FQg"

client = RESTClient(api_key=api_k)

def get_spy_historical(start_date="2024-01-01", end_date="2024-12-31"):
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


# ----------------------------------------------------------------
# 2️⃣ Fetch SPY Options Data
# ----------------------------------------------------------------
def get_spy_options(limit=50):
    """
    Fetches metadata about SPY option contracts (calls and puts).
    """
    results = []
    for opt in client.list_options_contracts(
        underlying_ticker="SPY",
        limit=limit,
        order="asc",
        sort="expiration_date"
    ):
        results.append({
            "symbol": opt.ticker,
            "expiration_date": opt.expiration_date,
            "strike_price": opt.strike_price,
            "type": opt.contract_type,       # 'call' or 'put'
            "underlying_ticker": opt.underlying_ticker,
            "exercise_style": opt.exercise_style,
            "primary_exchange": opt.primary_exchange
        })

    df = pd.DataFrame(results)
    return df

# -------------------------------------------------------------
# 3️⃣ Fetch Greeks + IV for each Option Contract
# -------------------------------------------------------------
def get_option_greeks(options_df):
    """
    For each option in options_df, fetch Greeks and IV using get_snapshot_option().
    """
    if options_df is None or options_df.empty:
        print("⚠️ Skipping Greeks fetch — options_df is empty.")
        return pd.DataFrame()

    data = []
    for _, row in options_df.iterrows():
        symbol = row["symbol"]
        try:
            snap = client.get_snapshot_option("SPY", symbol)
            if snap and snap.greeks:
                data.append({
                    "symbol": symbol,
                    "delta": snap.greeks.delta,
                    "gamma": snap.greeks.gamma,
                    "theta": snap.greeks.theta,
                    "vega": snap.greeks.vega,
                    "implied_volatility": snap.implied_volatility,
                    "last_price": snap.last_quote.p if snap.last_quote else None,
                    "underlying_price": snap.underlying_asset.last_trade.p if snap.underlying_asset else None
                })
            else:
                data.append({
                    "symbol": symbol,
                    "delta": None,
                    "gamma": None,
                    "theta": None,
                    "vega": None,
                    "implied_volatility": None,
                    "last_price": None,
                    "underlying_price": None
                })
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            continue

        time.sleep(0.25)

    return pd.DataFrame(data)


def read_csv_with_csv_module(file_path):
    """
    Reads a CSV file and returns a pandas DataFrame.
    """
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)

        rows = []
        for row in csv_reader:
            rows.append(row)

    df = pd.DataFrame(rows, columns=header)
    print(f"✅ Loaded {len(df)} rows from {file_path}")
    return df

# -------------------------------------------------------------
# 2️⃣ Fetch SPY Historical Data (for volatility estimate)
# -------------------------------------------------------------
def get_spy_historical(start_date="2024-01-01", end_date="2024-12-31"):
    data = []
    for bar in client.list_aggs(
        "SPY",
        1,
        "day",
        start_date,
        end_date,
        adjusted=True,
        sort="asc",
        limit=50000
    ):
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

# -------------------------------------------------------------
# 3️⃣ Estimate Historical Volatility
# -------------------------------------------------------------
def estimate_volatility(spy_df):
    # Remove any extra whitespace and convert to float
    spy_df['close'] = pd.to_numeric(spy_df['close'], errors='coerce')

    # Drop any rows where close is NaN after conversion
    spy_df = spy_df.dropna(subset=['close'])

    # Compute daily returns
    spy_df['returns'] = spy_df['close'].pct_change()

    # Estimate annualized volatility
    sigma = spy_df['returns'].std() * np.sqrt(252)
    print(f"📈 Estimated annual volatility: {sigma:.2%}")

    return sigma

# -------------------------------------------------------------
# 4️⃣ Black-Scholes Formula for Greeks
# -------------------------------------------------------------
def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": None, "gamma": None, "theta": None, "vega": None, "rho": None}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r*T) * norm.cdf(d2)
        rho = K * T * np.exp(-r*T) * norm.cdf(d2)
    else:  # put
        delta = -norm.cdf(-d1)
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r*T) * norm.cdf(-d2)
        rho = -K * T * np.exp(-r*T) * norm.cdf(-d2)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta / 365,
        "vega": vega / 100,
        "rho": rho / 100
    }

# -------------------------------------------------------------
# 6️⃣ Compute Greeks for Each Option
# -------------------------------------------------------------
def compute_greeks_for_options(options_df, spy_price, sigma, r=0.05):
    results = []
    today = datetime.now()

    for _, row in options_df.iterrows():
        try:
            exp_date = datetime.strptime(str(row['expiration_date'])[:10], '%Y-%m-%d')
            T = max((exp_date - today).days / 365, 1/365)
            greeks = black_scholes_greeks(
                S=spy_price,
                K=float(row['strike_price']),
                T=T,
                r=r,
                sigma=sigma,
                option_type=row['type']
            )
            results.append({**row, **greeks})
        except Exception as e:
            print(f"Error computing greeks for {row['symbol']}: {e}")
            continue

    return pd.DataFrame(results)

# -------------------------------------------------------------
# 7️⃣ Get Latest SPY Price
# -------------------------------------------------------------
def get_latest_spy_price():
    """
    Fetch latest SPY price using polygon-api-client v1.16.3
    """
    try:
        snap = client.get_snapshot_ticker("SPY")  # works in your version
        # last_trade is a dict
        if snap and hasattr(snap, 'last_trade') and snap.last_trade:
            price = snap.last_trade['p']  # price of last trade
        else:
            raise ValueError("No last trade data available in snapshot")
    except Exception as e:
        print(f"⚠️ Could not fetch snapshot: {e}")
        # fallback: get most recent daily aggregate
        agg = next(client.list_aggs("SPY", 1, "day", "2025-11-07", "2025-11-07", sort="desc", limit=1))
        price = agg.close
        print("⚠️ Using fallback aggregate price.")

    print(f"💰 Latest SPY price: {price}")
    return price



# ----------------------------------------------------------------
# 3️⃣ Run Example
# ----------------------------------------------------------------
if __name__ == "__main__":
    # print("Fetching SPY historical data...")
    # spy_df = get_spy_historical("2025-01-01", "2025-11-07")
    # print(spy_df.head(), "\n")

    # print("Fetching SPY options chain data...")
    # options_df = get_spy_options(limit=30)
    # print(options_df.head(), "\n")

    # spy_df.to_csv("spy_historical.csv", index=False)
    # options_df.to_csv("spy_options.csv", index=False)

    spy_df = read_csv_with_csv_module('spy_historical.csv')
    options_df = read_csv_with_csv_module('spy_options.csv')

    print("⚙️ Computing theoretical Greeks...")
    spy_price = get_latest_spy_price()
    sigma = estimate_volatility(spy_df)
    options_full_df = compute_greeks_for_options(options_df, spy_price, sigma)

    # Save for modeling
    #spy_df.to_csv("spy_historical.csv", index=False)
    options_full_df.to_csv("spy_options_with_greeks.csv", index=False)
    
    print("✅ Data saved to spy_historical.csv and spy_options.csv")