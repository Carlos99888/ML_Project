# import pandas as pd
# import pandas_ta as ta

# # Example: load your historical SPY data
# spy_df = pd.read_csv("spy_historical.csv")
# spy_df = spy_df.sort_values("date").reset_index(drop=True)


# # Simple and Exponential Moving Averages
# spy_df["SMA_20"] = ta.sma(spy_df["close"], length=20)
# spy_df["EMA_20"] = ta.ema(spy_df["close"], length=20)
# spy_df["EMA_50"] = ta.ema(spy_df["close"], length=50)
# spy_df["EMA_200"] = ta.ema(spy_df["close"], length=200)

# #RSI relative strength index
# #RSI tells you if SPY is overbought (>70) or oversold (<30).
# spy_df["RSI_14"] = ta.rsi(spy_df["close"], length=14)

# #MACD moving average convergence divergence
# macd = ta.macd(spy_df["close"], fast=12, slow=26, signal=9)
# spy_df = pd.concat([spy_df, macd], axis=1)

# #bollinger bands, ddds upper/lower bands and midline — captures volatility and price deviation.
# bbands = ta.bbands(spy_df["close"], length=20, std=2)
# spy_df = pd.concat([spy_df, bbands], axis=1)

# #ADX average directional index, measures trend strength (values above 25 = strong trend)
# spy_df["ADX_14"] = ta.adx(spy_df["high"], spy_df["low"], spy_df["close"], length=14)["ADX_14"]

# #volume-based indicators, these capture buying/selling pressure from volume trends.
# spy_df["OBV"] = ta.obv(spy_df["close"], spy_df["volume"])     # On-Balance Volume
# spy_df["CMF"] = ta.cmf(spy_df["high"], spy_df["low"], spy_df["close"], spy_df["volume"])  # Chaikin Money Flow



import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime

# -------------------------------------------------------------
# 1️⃣ Load Historical SPY Data
# -------------------------------------------------------------
def load_spy_data(file_path="spy_historical.csv"):
    spy_df = pd.read_csv(file_path)

    # Expect 'date' column (format: YYYY-MM-DD)
    if "date" not in spy_df.columns:
        raise ValueError("❌ Missing 'date' column in historical data file.")
    spy_df["date"] = pd.to_datetime(spy_df["date"])

    # Convert numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        spy_df[col] = pd.to_numeric(spy_df[col], errors="coerce")

    spy_df = spy_df.sort_values("date").dropna().reset_index(drop=True)
    print(f"✅ Loaded SPY historical data: {spy_df.shape[0]} rows")
    return spy_df


# -------------------------------------------------------------
# 2️⃣ Compute Technical Indicators
# -------------------------------------------------------------
def add_technical_indicators(df):
    print("⚙️ Adding technical indicators...")

    # Trend Indicators
    df["EMA_20"] = ta.ema(df["close"], length=20)
    df["EMA_50"] = ta.ema(df["close"], length=50)
    df["EMA_200"] = ta.ema(df["close"], length=200)
    df["SMA_20"] = ta.sma(df["close"], length=20)
    df["ADX_14"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]

    # Momentum Indicators
    df["RSI_14"] = ta.rsi(df["close"], length=14)

    # MACD
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)

    # Bollinger Bands
    bbands = ta.bbands(df["close"], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)

    # Volume-Based
    df["OBV"] = ta.obv(df["close"], df["volume"])
    df["CMF"] = ta.cmf(df["high"], df["low"], df["close"], df["volume"])

    # Returns & Volatility
    df["return_1d"] = df["close"].pct_change()
    df["return_5d"] = df["close"].pct_change(5)
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # Drop NaN rows caused by rolling calculations
    df = df.dropna().reset_index(drop=True)
    print(f"✅ Added indicators — total columns: {df.shape[1]}")
    return df


# -------------------------------------------------------------
# 3️⃣ Load Options Data
# -------------------------------------------------------------
def load_options_data(file_path="spy_options_with_greeks.csv"):
    options_df = pd.read_csv(file_path)

    if "expiration_date" not in options_df.columns:
        raise ValueError("❌ Missing 'expiration_date' column in options data file.")
    options_df["expiration_date"] = pd.to_datetime(options_df["expiration_date"])

    options_df = options_df.sort_values("expiration_date").reset_index(drop=True)
    print(f"✅ Loaded options data: {options_df.shape[0]} rows")
    return options_df


# -------------------------------------------------------------
# 4️⃣ Merge SPY + Options Data
# -------------------------------------------------------------
def merge_datasets(spy_df, options_df):
    print("🔄 Merging options data (expiration_date) with SPY indicators (date)...")

    merged_df = pd.merge_asof(
        options_df.sort_values("expiration_date"),
        spy_df.sort_values("date"),
        left_on="expiration_date",
        right_on="date",
        direction="backward"  # get last available price before option expiry
    )

    print(f"✅ Merged dataset shape: {merged_df.shape}")
    return merged_df


# -------------------------------------------------------------
# 5️⃣ Run Example
# -------------------------------------------------------------
if __name__ == "__main__":
    spy_df = load_spy_data("spy_historical.csv")
    spy_df = add_technical_indicators(spy_df)

    options_df = load_options_data("spy_options_with_greeks.csv")

    merged_df = merge_datasets(spy_df, options_df)

    merged_df.to_csv("spy_training_dataset.csv", index=False)
    print("💾 Saved merged dataset to spy_training_dataset.csv")
    print(merged_df.head())
