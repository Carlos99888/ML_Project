import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA

# -------------------------------------------------------------
# 1️⃣ Load Historical SPY Data
# -------------------------------------------------------------
def load_spy_data(file_path="spy_historical.csv"):
    """
    Loads SPY historical OHLCV data from CSV.
    Expected columns: date, open, high, low, close, volume
    Converts 'date' to datetime and renames columns for compatibility
    with the backtesting library.
    """
    df = pd.read_csv(file_path)
    if "date" not in df.columns:
        raise ValueError("❌ Missing 'date' column in CSV file")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Rename columns for compatibility
    df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }, inplace=True)

    # Use Date as index (important for plotting & time-based features)
    df.set_index("Date", inplace=True)
    return df


# -------------------------------------------------------------
# 2️⃣ Add Technical Indicators
# -------------------------------------------------------------
def add_indicators(df):
    """
    Adds a variety of commonly used technical indicators to the DataFrame.
    These are useful features for both ML models and trading strategy design.
    """

    # -------------------------
    # 🔹 TREND INDICATORS
    # -------------------------
    # Exponential Moving Averages (EMA)
    # - Shorter EMAs (20) react quickly to price changes.
    # - Longer EMAs (50, 200) show longer-term trends.
    df["EMA_20"] = ta.ema(df["Close"], length=20)
    df["EMA_50"] = ta.ema(df["Close"], length=50)
    df["EMA_200"] = ta.ema(df["Close"], length=200)

    # -------------------------
    # 🔹 MOMENTUM INDICATORS
    # -------------------------
    # RSI (Relative Strength Index)
    # - Ranges from 0–100.
    # - RSI > 70 = overbought, RSI < 30 = oversold.
    df["RSI_14"] = ta.rsi(df["Close"], length=14)

    # -------------------------
    # 🔹 MACD (Moving Average Convergence Divergence)
    # -------------------------
    # - MACD shows momentum and trend strength.
    # - Consists of: MACD line, signal line, histogram.
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)

    # -------------------------
    # 🔹 VOLATILITY INDICATORS
    # -------------------------
    # Bollinger Bands (BBL, BBM, BBU)
    # - Upper/lower bands measure price volatility.
    # - Price touching the bands can indicate reversal points.
    bbands = ta.bbands(df["Close"], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)

    # -------------------------
    # 🔹 VOLUME-BASED INDICATORS
    # -------------------------
    # OBV (On-Balance Volume)
    # - Cumulative volume indicator showing whether volume confirms trends.
    df["OBV"] = ta.obv(df["Close"], df["Volume"])

    # CMF (Chaikin Money Flow)
    # - Combines price and volume to measure buying/selling pressure.
    df["CMF"] = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"])

    # -------------------------
    # 🔹 RETURNS & VOLATILITY METRICS
    # -------------------------
    # 1-day and 5-day returns (price momentum)
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)

    # 20-day rolling volatility (standard deviation of returns)
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # Drop NaN rows caused by rolling window calculations
    df = df.fillna(method="bfill").fillna(method="ffill")
    return df


# -------------------------------------------------------------
# 3️⃣ Example Visualization Strategy
# -------------------------------------------------------------
class SmaCross(Strategy):
    """
    Simple Moving Average Crossover Strategy
    (for visualization purposes only)
    - Buys when short-term SMA crosses above long-term SMA.
    - Sells when it crosses below.
    """
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
# 4️⃣ Main Run
# -------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Load data
    spy_df = load_spy_data("spy_historical.csv")

    # Step 2: Compute indicators
    spy_df = add_indicators(spy_df)

    # Step 3: Save dataset for machine learning / deep learning
    spy_df.to_csv("spy_training_dataset_v3.csv")
    print("💾 Saved enhanced dataset → spy_training_dataset.csv")

    # Step 4: Display interactive candlestick chart with SMA crossover
    bt = Backtest(spy_df, SmaCross, cash=100_000, commission=0.002)
    results = bt.run()
    bt.plot(open_browser=True)
