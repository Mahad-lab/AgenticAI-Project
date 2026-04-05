"""
tools/technical_tools.py — Fibonacci, RSI, Moving Averages, Elliott Wave (simplified)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from config import (
    FIB_LEVELS, BUY_ZONE_LOW, BUY_ZONE_HIGH, STOP_LOSS_FIB,
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    MA_SHORT, MA_LONG, VOLUME_MA_PERIOD, SWING_WINDOW,
)
from tools.data_tools import fetch_ohlcv


# ─── Swing Point Detection ────────────────────────────────────────────────────

def detect_swings(df: pd.DataFrame, window: int = SWING_WINDOW
                  ) -> tuple[pd.Series, pd.Series]:
    """
    Identify swing highs and swing lows using a rolling-window approach.

    A swing high at bar i  : High[i] is the max in [i-window, i+window]
    A swing low  at bar i  : Low[i]  is the min in [i-window, i+window]

    Returns two boolean Series (swing_highs, swing_lows).
    """
    highs = df["High"]
    lows  = df["Low"]
    n     = len(df)

    swing_high = pd.Series(False, index=df.index)
    swing_low  = pd.Series(False, index=df.index)

    for i in range(window, n - window):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        if highs.iloc[i] == highs.iloc[lo:hi].max():
            swing_high.iloc[i] = True
        if lows.iloc[i] == lows.iloc[lo:hi].min():
            swing_low.iloc[i] = True

    return swing_high, swing_low


def last_significant_swing(df: pd.DataFrame,
                            window: int = SWING_WINDOW
                            ) -> tuple[float, float, int, int]:
    """
    Return the most recent significant swing low and swing high
    that form a clear retracement structure.

    Returns: (swing_low_price, swing_high_price, low_idx, high_idx)
    If the high comes AFTER the low  → upswing  (retracement is pullback)
    If the low  comes AFTER the high → downswing (retracement is bounce)
    """
    sh, sl = detect_swings(df, window)
    sh_prices = df["High"][sh].dropna()
    sl_prices = df["Low"][sl].dropna()

    if sh_prices.empty or sl_prices.empty:
        # Fallback: use rolling max/min over last 60 bars
        recent = df.tail(60)
        return (float(recent["Low"].min()),
                float(recent["High"].max()),
                int(recent["Low"].idxmin().value),
                int(recent["High"].idxmax().value))

    # Most recent swing high and low
    last_high_idx = sh_prices.index[-1]
    last_low_idx  = sl_prices.index[-1]

    return (
        float(df.loc[last_low_idx,  "Low"]),
        float(df.loc[last_high_idx, "High"]),
        df.index.get_loc(last_low_idx),
        df.index.get_loc(last_high_idx),
    )


# ─── Fibonacci / Bisection / Trisection ─────────────────────────────────────

def calculate_fib_levels(df: pd.DataFrame,
                          swing_low:  float | None = None,
                          swing_high: float | None = None,
                          ) -> dict[str, float]:
    """
    Calculate Fibonacci retracement levels including bisection (50%)
    and trisection (33.3% / 66.7%) over the most recent swing.

    Retracement formula (upswing → pullback):
        level_price = swing_high - ratio * (swing_high - swing_low)
    """
    if swing_low is None or swing_high is None:
        swing_low, swing_high, _, _ = last_significant_swing(df)

    diff   = swing_high - swing_low
    levels = {}
    for label, ratio in FIB_LEVELS.items():
        price = swing_high - ratio * diff   # retracement from high
        levels[label] = round(price, 4)

    levels["swing_low"]  = round(swing_low,  4)
    levels["swing_high"] = round(swing_high, 4)
    levels["diff"]       = round(diff,        4)
    return levels


# ─── RSI ──────────────────────────────────────────────────────────────────────

def calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI on a price series."""
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.round(2)


# ─── Moving Averages & Volume ─────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MAs, volume MA in-place (returns new copy)."""
    df = df.copy()
    df["RSI"]       = calculate_rsi(df["Close"])
    df["MA_short"]  = df["Close"].rolling(MA_SHORT).mean().round(4)
    df["MA_long"]   = df["Close"].rolling(MA_LONG).mean().round(4)
    df["Vol_MA"]    = df["Volume"].rolling(VOLUME_MA_PERIOD).mean()
    df["Vol_Ratio"] = (df["Volume"] / df["Vol_MA"]).round(2)  # >1.5 = high volume
    return df


# ─── Simplified Elliott Wave Trend Detection ─────────────────────────────────

def detect_wave_structure(df: pd.DataFrame) -> dict:
    """
    Simplified Elliott Wave-inspired trend detection.

    Approach:
      1. Identify the last 5 significant swing points.
      2. If the pattern is Higher-Highs + Higher-Lows → impulse wave (bullish)
         If the pattern is Lower-Highs + Lower-Lows  → impulse wave (bearish)
         Otherwise → corrective / sideways
      3. Estimate which wave we might be in based on position relative
         to the overall range.

    Returns a dict with: trend, wave_position, confidence, description
    """
    sh_mask, sl_mask = detect_swings(df, SWING_WINDOW)
    swing_highs = df["High"][sh_mask]
    swing_lows  = df["Low"][sl_mask]

    result = {
        "trend":         "UNDETERMINED",
        "wave_position": "UNKNOWN",
        "confidence":    "LOW",
        "description":   "",
    }

    # Need at least 3 swing highs and lows
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        result["description"] = "Insufficient swing points for wave analysis."
        return result

    recent_highs = swing_highs.tail(3).values
    recent_lows  = swing_lows.tail(3).values

    hh = all(recent_highs[i] < recent_highs[i+1] for i in range(len(recent_highs)-1))
    hl = all(recent_lows[i]  < recent_lows[i+1]  for i in range(len(recent_lows)-1))
    lh = all(recent_highs[i] > recent_highs[i+1] for i in range(len(recent_highs)-1))
    ll = all(recent_lows[i]  > recent_lows[i+1]  for i in range(len(recent_lows)-1))

    close_now   = df["Close"].iloc[-1]
    period_high = df["High"].max()
    period_low  = df["Low"].min()
    range_pct   = ((close_now - period_low) / (period_high - period_low)
                   if period_high != period_low else 0.5)

    if hh and hl:
        result["trend"]      = "BULLISH_IMPULSE"
        result["confidence"] = "MEDIUM"
        if range_pct > 0.75:
            result["wave_position"] = "WAVE_5_OR_CORRECTION"
            result["description"]   = (
                "Higher-highs + higher-lows (bullish impulse). "
                "Price near recent highs — possible Wave 5 completion "
                "or start of ABC correction. Exercise caution on new longs."
            )
        elif range_pct > 0.4:
            result["wave_position"] = "WAVE_3_OR_4"
            result["description"]   = (
                "Bullish structure intact. Mid-range position suggests Wave 3/4. "
                "Retracement zones (50–61.8%) may offer good entry if pullback occurs."
            )
        else:
            result["wave_position"] = "WAVE_1_OR_2"
            result["description"]   = (
                "Early-stage bullish impulse. Price near lows — could be Wave 1 "
                "or Wave 2 correction. Watch for confirmation before entering."
            )
    elif lh and ll:
        result["trend"]      = "BEARISH_IMPULSE"
        result["confidence"] = "MEDIUM"
        result["wave_position"] = "DOWNTREND"
        result["description"]   = (
            "Lower-highs + lower-lows confirm bearish impulse wave. "
            "Avoid buying into rallies unless at key support with strong RSI divergence."
        )
    else:
        result["trend"]      = "CORRECTIVE_SIDEWAYS"
        result["confidence"] = "LOW"
        result["wave_position"] = "ABC_CORRECTION"
        result["description"]   = (
            "Mixed swing structure suggests ABC corrective phase or "
            "sideways consolidation. Key Fib levels (38.2–61.8%) should act "
            "as support/resistance."
        )

    return result


# ─── Buy Signal Logic ─────────────────────────────────────────────────────────

def generate_signal(df: pd.DataFrame) -> dict:
    """
    Combine Fibonacci retracement + RSI to generate a trade signal.

    Rules:
      BUY  : price is between 50–61.8% Fib AND RSI < RSI_OVERSOLD
               AND volume ratio > 1.0 (above-average volume)
      SELL : price near 23.6% Fib (take profit) OR RSI > RSI_OVERBOUGHT
      WATCH: price between 38.2–50% Fib (approaching buy zone)
      WAIT : otherwise
    """
    df_ind = add_indicators(df)
    fib    = calculate_fib_levels(df)
    wave   = detect_wave_structure(df)

    close      = float(df_ind["Close"].iloc[-1])
    rsi        = float(df_ind["RSI"].dropna().iloc[-1])
    vol_ratio  = float(df_ind["Vol_Ratio"].dropna().iloc[-1])
    ma_short   = float(df_ind["MA_short"].dropna().iloc[-1])
    ma_long    = float(df_ind["MA_long"].dropna().iloc[-1]) if len(df) >= MA_LONG else None

    level_618  = fib["61.8"]
    level_50   = fib["50.0"]
    level_382  = fib["38.2"]
    level_236  = fib["23.6"]
    level_786  = fib["78.6"]
    stop_loss  = level_786

    in_buy_zone   = level_618 <= close <= level_50
    near_tp       = close >= level_236
    approaching   = level_50 < close <= level_382

    # Volume confirmation
    vol_ok = vol_ratio >= 1.0

    if in_buy_zone and rsi < RSI_OVERSOLD and vol_ok:
        signal = "BUY"
        reason = (f"Price ({close:.2f}) is in the 50–61.8% Fib buy zone "
                  f"({level_618:.2f}–{level_50:.2f}), RSI={rsi:.1f} (oversold), "
                  f"volume {vol_ratio:.1f}x average.")
    elif in_buy_zone and rsi < RSI_OVERSOLD:
        signal = "BUY_WEAK"
        reason = (f"Price in Fib buy zone but volume below average ({vol_ratio:.1f}x). "
                  f"RSI={rsi:.1f}. Wait for volume confirmation.")
    elif near_tp or rsi > RSI_OVERBOUGHT:
        signal = "SELL/TAKE_PROFIT"
        reason = (f"Price ({close:.2f}) near 23.6% Fib target ({level_236:.2f}) "
                  f"or RSI={rsi:.1f} (overbought).")
    elif approaching:
        signal = "WATCH"
        reason = (f"Price ({close:.2f}) approaching buy zone ({level_618:.2f}–{level_50:.2f}). "
                  f"Set alert. RSI={rsi:.1f}.")
    else:
        signal = "WAIT"
        reason = (f"Price ({close:.2f}) outside key Fib zones. "
                  f"RSI={rsi:.1f}. No clear setup.")

    return {
        "signal":       signal,
        "reason":       reason,
        "close":        round(close, 2),
        "rsi":          round(rsi, 2),
        "vol_ratio":    round(vol_ratio, 2),
        "fib_levels":   {k: v for k, v in fib.items()
                         if k not in ("diff", "swing_low", "swing_high")},
        "stop_loss":    round(stop_loss, 2),
        "take_profit":  round(level_236, 2),
        "wave":         wave,
        "ma_short":     round(ma_short, 2),
        "ma_long":      round(ma_long, 2) if ma_long else "N/A",
        "ma_signal":    ("GOLDEN_CROSS" if ma_long and ma_short > ma_long
                         else "DEATH_CROSS" if ma_long and ma_short < ma_long
                         else "N/A"),
    }


# ─── LangChain Tool ───────────────────────────────────────────────────────────

@tool
def analyze_technical(query: str) -> str:
    """
    Run full technical analysis on a PSX stock.

    Input (plain string):  "TICKER"  or  "TICKER,DAYS"
    Examples:
        "WTL"
        "KEL,180"

    Returns:
      - Fibonacci retracement levels (50%, 61.8%, etc.)
      - Bisection (50%) and trisection (33%/66%) levels
      - RSI, MA signals
      - Elliott Wave structure assessment
      - BUY / SELL / WATCH / WAIT signal with reasoning
      - Suggested stop-loss and take-profit levels
    """
    parts  = [p.strip().upper() for p in query.split(",")]
    ticker = parts[0]
    days   = int(parts[1]) if len(parts) > 1 else 365

    try:
        df  = fetch_ohlcv(ticker, days=days)
        sig = generate_signal(df)
    except Exception as exc:
        return f"Technical analysis error for {ticker}: {exc}"

    fib = sig["fib_levels"]
    wave = sig["wave"]

    lines = [
        f"🔍  Technical Analysis — {ticker}  ({days}d)",
        "",
        f"Signal    : {sig['signal']}",
        f"Reasoning : {sig['reason']}",
        "",
        "── Fibonacci / Bisection / Trisection Levels ──",
    ]
    for label, price in sorted(fib.items(), key=lambda x: float(x[0])):
        marker = ""
        val    = float(label)
        if val == 50.0:  marker = " ← bisection"
        if val in (33.3, 66.7): marker = " ← trisection"
        if val == 61.8:  marker = " ← 🟢 key buy zone"
        lines.append(f"  {label:>6}% → {price:.4f} PKR{marker}")

    lines += [
        "",
        f"Stop Loss    : {sig['stop_loss']:.4f} PKR  (78.6% Fib)",
        f"Take Profit  : {sig['take_profit']:.4f} PKR  (23.6% Fib)",
        "",
        "── Indicators ──",
        f"RSI ({RSI_PERIOD}): {sig['rsi']}  "
            f"{'(OVERSOLD ✅)' if sig['rsi'] < RSI_OVERSOLD else '(OVERBOUGHT ⚠️)' if sig['rsi'] > RSI_OVERBOUGHT else '(NEUTRAL)'}",
        f"MA{MA_SHORT}   : {sig['ma_short']} PKR",
        f"MA{MA_LONG}   : {sig['ma_long']} PKR",
        f"MA Signal : {sig['ma_signal']}",
        f"Vol Ratio : {sig['vol_ratio']}x  (>1.5 = high volume)",
        "",
        "── Elliott Wave (Simplified) ──",
        f"Trend     : {wave['trend']}",
        f"Position  : {wave['wave_position']}",
        f"Confidence: {wave['confidence']}",
        f"Notes     : {wave['description']}",
    ]

    return "\n".join(lines)
