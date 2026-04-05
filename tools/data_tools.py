"""
tools/data_tools.py — PSX data fetching via the `psx` pip package
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, date

import pandas as pd

from langchain_core.tools import tool


# ─── Raw fetch (importable without LangChain) ────────────────────────────────

def fetch_ohlcv(ticker: str, days: int = 365) -> pd.DataFrame:
    """
    Fetch OHLCV data from PSX using the `psx` package.

    Returns a DataFrame with DatetimeIndex and columns:
        Open, High, Low, Close, Volume
    Sorted ascending by date, NaN rows dropped.
    """
    try:
        from psx import stocks as psx_stocks
    except ImportError:
        raise ImportError("Install psx with:  pip install psx")

    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=days)

    df = psx_stocks(ticker, start=start_dt.date(), end=end_dt.date())

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. "
                         "Check the symbol (e.g. 'WTL', 'KEL') and ensure "
                         "it is listed on PSX.")

    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna()

    # Normalise column names (psx sometimes returns title-cased names)
    df.columns = [c.strip().title() for c in df.columns]
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected columns from psx for '{ticker}': "
                         f"{df.columns.tolist()}  — missing {missing}")

    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def get_current_price(ticker: str) -> float:
    """Return the latest closing price for a ticker."""
    df = fetch_ohlcv(ticker, days=7)
    return float(df["Close"].iloc[-1])


# ─── LangChain Tool ───────────────────────────────────────────────────────────

@tool
def get_stock_summary(query: str) -> str:
    """
    Fetch a summary of recent trading data for a PSX stock.

    Input (plain string):  "TICKER"  or  "TICKER,DAYS"
    Examples:
        "WTL"
        "KEL,180"

    Returns a text summary with: latest price, recent high/low,
    average volume, % change over period, and data date range.
    """
    parts  = [p.strip().upper() for p in query.split(",")]
    ticker = parts[0]
    days   = int(parts[1]) if len(parts) > 1 else 90

    try:
        df = fetch_ohlcv(ticker, days=days)
    except Exception as exc:
        return f"Error fetching data for {ticker}: {exc}"

    latest        = df["Close"].iloc[-1]
    period_high   = df["High"].max()
    period_low    = df["Low"].min()
    avg_vol       = df["Volume"].mean()
    pct_change    = ((df["Close"].iloc[-1] - df["Close"].iloc[0])
                     / df["Close"].iloc[0] * 100)
    recent_5d_avg = df["Close"].tail(5).mean()
    date_start    = df.index[0].strftime("%Y-%m-%d")
    date_end      = df.index[-1].strftime("%Y-%m-%d")
    bars          = len(df)

    trend = "UPTREND" if df["Close"].tail(10).is_monotonic_increasing else (
            "DOWNTREND" if df["Close"].tail(10).is_monotonic_decreasing else "SIDEWAYS")

    summary = {
        "ticker":           ticker,
        "data_range":       f"{date_start} → {date_end}  ({bars} trading days)",
        "latest_close_pkr": round(latest, 2),
        "5d_avg_close_pkr": round(recent_5d_avg, 2),
        "period_high_pkr":  round(period_high, 2),
        "period_low_pkr":   round(period_low, 2),
        "period_change_pct":f"{pct_change:+.2f}%",
        "avg_daily_volume": f"{avg_vol:,.0f}",
        "last_volume":      f"{df['Volume'].iloc[-1]:,.0f}",
        "recent_trend":     trend,
    }

    lines = [f"📊  Stock Summary — {ticker}  ({days}d)"]
    lines += [f"  {k}: {v}" for k, v in summary.items() if k != "ticker"]
    return "\n".join(lines)
