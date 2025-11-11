# stock_class.py

from polygon import RESTClient
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA

import pandas as pd
import pandas_ta as ta
import numpy as np
import time
from datetime import datetime
from scipy.stats import norm
import csv


class StockData:
    """
    Class for fetching, analyzing, and visualizing SPY historical and options data.
    Integrates:
      - Polygon.io API data (historical + options + Greeks)
      - Black–Scholes model for theoretical Greeks
      - Technical indicators for ML/deep learning
      - Backtesting visualization (SMA crossover)
    """

    def __init__(self, api_key: str):
        self.client = RESTClient(api_key=api_key)
        print("✅ Polygon RESTClient initialized.")

    # -------------------------------------------------------------
    # 1️⃣ Fetch Historical SPY OHLCV
    # -------------------------------------------------------------
    def get_spy_historical(self, start_date="2024-01-01", end_date="2024-12-31"):
        data = []
        for bar in self.client.list_aggs(
            ticker="SPY", 
            multiplier=1, 
            timespan="day", 
            from_=start_date, 
            to=end_date,
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
        print(f"📈 Loaded {len(df)} daily SPY bars.")
        return df

    # -------------------------------------------------------------
    # 2️⃣ Fetch SPY Options Contracts
    # -------------------------------------------------------------
    def get_spy_options(self, limit=50):
        results = []
        for opt in self.client.list_options_contracts(
            underlying_ticker="SPY", limit=limit, order="asc", sort="expiration_date"
        ):
            results.append({
                "symbol": opt.ticker,
                "expiration_date": opt.expiration_date,
                "strike_price": opt.strike_price,
                "type": opt.contract_type,
                "exercise_style": opt.exercise_style,
                "primary_exchange": opt.primary_exchange
            })
        df = pd.DataFrame(results)
        print(f"📄 Retrieved {len(df)} SPY options contracts.")
        return df

    # -------------------------------------------------------------
    # 3️⃣ Fetch Greeks + IV from API
    # -------------------------------------------------------------
    def get_option_greeks(self, options_df):
        if options_df is None or options_df.empty:
            print("⚠️ Skipping Greeks fetch — no option data.")
            return pd.DataFrame()

        data = []
        for _, row in options_df.iterrows():
            symbol = row["symbol"]
            try:
                snap = self.client.get_snapshot_option("SPY", symbol)
                if snap and snap.greeks:
                    data.append({
                        "symbol": symbol,
                        "delta": snap.greeks.delta,
                        "gamma": snap.greeks.gamma,
                        "theta": snap.greeks.theta,
                        "vega": snap.greeks.vega,
                        "implied_volatility": snap.implied_volatility
                    })
            except Exception as e:
                print(f"⚠️ Error fetching {symbol}: {e}")
                continue
            time.sleep(0.25)
        df = pd.DataFrame(data)
        print(f"✅ Retrieved Greeks for {len(df)} options.")
        return df

    # -------------------------------------------------------------
    # 4️⃣ Read CSV Utility
    # -------------------------------------------------------------
    @staticmethod
    def read_csv(file_path):
        with open(file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        df = pd.DataFrame(rows, columns=header)
        print(f"📥 Loaded {len(df)} rows from {file_path}")
        return df

    # -------------------------------------------------------------
    # 5️⃣ Estimate Historical Volatility
    # -------------------------------------------------------------
    @staticmethod
    def estimate_volatility(spy_df):
        spy_df['close'] = pd.to_numeric(spy_df['close'], errors='coerce')
        spy_df.dropna(subset=['close'], inplace=True)
        spy_df['returns'] = spy_df['close'].pct_change()
        sigma = spy_df['returns'].std() * np.sqrt(252)
        print(f"📊 Estimated annual volatility: {sigma:.2%}")
        return sigma

    # -------------------------------------------------------------
    # 6️⃣ Black-Scholes Greeks
    # -------------------------------------------------------------
    @staticmethod
    def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return {"delta": None, "gamma": None, "theta": None, "vega": None, "rho": None}

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            delta = norm.cdf(d1)
            theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r*T) * norm.cdf(d2)
            rho = K * T * np.exp(-r*T) * norm.cdf(d2)
        else:
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
    # 7️⃣ Compute Greeks for Options (theoretical)
    # -------------------------------------------------------------
    def compute_greeks_for_options(self, options_df, spy_price, sigma, r=0.05):
        results = []
        today = datetime.now()

        for _, row in options_df.iterrows():
            try:
                exp_date = datetime.strptime(str(row['expiration_date'])[:10], '%Y-%m-%d')
                T = max((exp_date - today).days / 365, 1/365)
                greeks = self.black_scholes_greeks(
                    S=spy_price,
                    K=float(row['strike_price']),
                    T=T,
                    r=r,
                    sigma=sigma,
                    option_type=row['type']
                )
                results.append({**row, **greeks})
            except Exception as e:
                print(f"Error computing {row['symbol']}: {e}")
        return pd.DataFrame(results)

    # -------------------------------------------------------------
    # 8️⃣ Latest SPY Price
    # -------------------------------------------------------------
    def get_latest_spy_price(self):
        try:
            snap = self.client.get_snapshot_ticker("SPY")
            price = snap.last_trade['p']
        except Exception as e:
            print(f"⚠️ Snapshot failed: {e}")
            agg = next(self.client.list_aggs("SPY", 1, "day", "2025-11-07", "2025-11-07", limit=1))
            price = agg.close
        print(f"💰 Latest SPY Price: {price}")
        return price

    # -------------------------------------------------------------
    # 9️⃣ Add Technical Indicators
    # -------------------------------------------------------------
    @staticmethod
    def add_indicators(df):
        df.rename(columns={
            "date": "Date", "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume"
        }, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        # Trend
        df["EMA_20"] = ta.ema(df["Close"], length=20)
        df["EMA_50"] = ta.ema(df["Close"], length=50)
        df["EMA_200"] = ta.ema(df["Close"], length=200)
        # Momentum
        df["RSI_14"] = ta.rsi(df["Close"], length=14)
        df = pd.concat([df, ta.macd(df["Close"])], axis=1)
        # Volatility
        df = pd.concat([df, ta.bbands(df["Close"])], axis=1)
        # Volume
        df["OBV"] = ta.obv(df["Close"], df["Volume"])
        df["CMF"] = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"])
        # Returns & Volatility
        df["return_1d"] = df["Close"].pct_change()
        df["return_5d"] = df["Close"].pct_change(5)
        df["volatility_20d"] = df["return_1d"].rolling(20).std()
        df = df.fillna(method="bfill").fillna(method="ffill")
        return df

    # -------------------------------------------------------------
    # 🔟 Visualization Strategy
    # -------------------------------------------------------------
    class SmaCross(Strategy):
        n1 = 20
        n2 = 50

        def init(self):
            price = self.data.Close
            self.sma1 = self.I(SMA, price, self.n1)
            self.sma2 = self.I(SMA, price, self.n2)

        def next(self):
            if crossover(self.sma1, self.sma2):
                self.buy()
            elif crossover(self.sma2, self.sma1):
                self.sell()

    # -------------------------------------------------------------
    # 11️⃣ Run Backtest Visualization
    # -------------------------------------------------------------
    @staticmethod
    def visualize(df):
        bt = Backtest(df, StockData.SmaCross, cash=100_000, commission=0.002)
        results = bt.run()
        print("📊 Strategy Summary:\n", results)
        bt.plot(open_browser=True)


# -------------------------------------------------------------
# ✅ Example Usage
# -------------------------------------------------------------
if __name__ == "__main__":
    api_key = "YOUR_POLYGON_API_KEY"
    stock = StockData(api_key)

    spy_df = stock.get_spy_historical("2025-01-01", "2025-11-07")
    options_df = stock.get_spy_options(limit=20)

    sigma = stock.estimate_volatility(spy_df)
    spy_price = stock.get_latest_spy_price()
    options_greeks = stock.compute_greeks_for_options(options_df, spy_price, sigma)

    spy_df = stock.add_indicators(spy_df)
    spy_df.to_csv("spy_training_dataset.csv")
    options_greeks.to_csv("spy_options_with_greeks.csv")
    print("💾 Saved training datasets successfully!")

    # Optional visualization
    stock.visualize(spy_df)
