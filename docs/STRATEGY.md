# Trading Strategy Guide

## Overview

The PSX Trading System implements a **Fibonacci retracement mean-reversion strategy**
designed specifically for the Pakistan Stock Exchange. It identifies pullbacks within
established trends using Fibonacci levels, confirmed by RSI and volume.

---

## Strategy Logic

### Entry Rules (BUY Signal)

All three conditions must be met:

```
1. PRICE: Close is in the 50.0%–61.8% Fibonacci retracement zone
2. RSI:   RSI(14) < 40 (oversold — relaxed threshold for PSX)
3. VOLUME: Volume ≥ 1.0× the 20-day average volume
```

#### Why these rules?

**50–61.8% Fib Zone:** In trending markets, pullbacks typically retrace to
the golden ratio zone. This is the most widely watched retracement area globally.

**RSI < 40:** Standard RSI oversold is 30, but PSX stocks are more volatile.
Relaxing to 40 captures more valid entry points while still filtering overbought.

**Volume ≥ 1× Average:** Confirms institutional interest in the pullback.
Low-volume pullbacks are more likely to continue declining.

### Exit Rules

| Condition | Action | Level |
|---|---|---|
| **Take Profit** | Sell | Price reaches 23.6% Fib level |
| **Stop Loss** | Sell | Price falls below 78.6% Fib level |
| **End of Data** | Sell | Last bar of available data |

#### Why 23.6% for TP?
The 23.6% level is the first Fib retracement level. In a mean-reversion strategy,
we don't need the price to return to the swing high — a partial recovery is sufficient
for a profitable trade. This increases win rate at the cost of smaller wins.

#### Why 78.6% for SL?
The 78.6% level is the deepest retracement level before a full reversal.
If price retraces this far, the trend structure is likely broken.

### Position Sizing (Risk-Based)

```
Risk Amount      = Current Capital × 2%
Risk Per Share   = Entry Price - Stop Loss Price
Number of Shares = floor(Risk Amount / Risk Per Share)

Max Position     = min(Calculated Shares, 50% of Capital in PKR / Entry Price)
```

**Example with 5,000 PKR capital:**
- Risk = 5,000 × 2% = 100 PKR max loss per trade
- If entry = 5.00 PKR and stop = 4.50 PKR (risk = 0.50/share)
- Shares = floor(100 / 0.50) = 200 shares
- Cost = 200 × 5.00 = 1,000 PKR (within 50% max = 2,500 PKR ✓)

---

## Fibonacci Level Details

The system calculates these retracement levels:

| Level | Label | Type | Purpose |
|---|---|---|---|
| 0.0% | Swing High | — | Trend extreme |
| 23.6% | TP Target | Gold | Take profit |
| 33.3% | Trisection | Teal | Secondary support/resistance |
| 38.2% | Secondary | Blue | Deep pullback zone |
| 50.0% | Bisection | Yellow | Midpoint + buy zone start |
| 61.8% | Golden Ratio | Green | **Key buy zone** |
| 66.7% | Trisection | Teal | Secondary support/resistance |
| 78.6% | Deep Retracement | Red | Stop loss zone |
| 100.0% | Swing Low | — | Trend extreme |

**Formula:**
```
Level Price = Swing High - (Ratio × (Swing High - Swing Low))
```

---

## Elliott Wave (Simplified)

The system does NOT implement full Elliott Wave theory (which requires significant
subjective judgment). Instead, it uses a simplified pattern detection:

### Detection Method
1. Find all significant swing highs and lows (local extrema over a ±5 bar window)
2. Check if last 3 highs and lows form a pattern

### Patterns

| Pattern | Highs | Lows | Interpretation |
|---|---|---|---|
| **Bullish Impulse** | Higher Highs | Higher Lows | Uptrend in progress |
| **Bearish Impulse** | Lower Highs | Lower Lows | Downtrend in progress |
| **Corrective/Sideways** | Mixed | Mixed | Consolidation phase |

### Position Estimation
Based on where price sits in the total range:
- **Near lows** (< 40% of range) → Early wave (1-2)
- **Mid-range** (40-75%) → Middle wave (3-4)
- **Near highs** (> 75%) → Late wave (5 or correction imminent)

---

## Backtesting Methodology

### Walk-Forward Approach
The backtester does NOT use in-sample/out-of-sample splits. Instead, it simulates
trading in chronological order, recalculating Fibonacci levels on a rolling 90-bar
window. This avoids look-ahead bias.

### Warm-up Period
First 60 bars are used for indicator initialization (RSI, MAs) and are not traded.

### Rolling Window
Fibonacci levels are recalculated every bar using the last 90 bars of data,
simulating how a trader would see the market in real-time.

### Commission
A flat 0.1% per side (0.2% round trip) is deducted from each trade,
matching typical PSX brokerage rates.

### Entry Validation
Before entering, the system checks:
- Stop loss is BELOW entry price
- Take profit is ABOVE entry price
- Enough capital to buy at least 1 share
- Position doesn't exceed 50% of capital

---

## Shariah Compliance Screening

### KMI-30 Index
The KMI-30 is a Shariah-compliant index consisting of 30 stocks that pass
Islamic finance screening criteria:
- No conventional banking, insurance, or leasing
- No interest-bearing debt > 30% of market cap
- No non-compliant income > 5% of total revenue

### KMI All-Share
A broader index covering all Shariah-compliant stocks on PSX.
The system screens against a maintained list of low-priced (< 10 PKR)
stocks from this index.

### Compliance Status
| Status | Meaning |
|---|---|
| **COMPLIANT** | Listed in KMI All-Share or KMI-30 |
| **NOT_COMPLIANT** | Conventional bank, insurance, or known non-compliant |
| **UNVERIFIED** | Not in local database — check PSX website |

---

## Risk Management Rules

Always enforced by the system:

1. **Max risk per trade:** 2% of current capital
2. **Max single position:** 50% of capital
3. **Always use stop loss** (at 78.6% Fib)
4. **Shariah check before recommending** any trade
5. **Warning if stock is not KMI-compliant**

---

## Parameter Reference

All parameters are configurable in `config.py`:

| Parameter | Default | Range | Purpose |
|---|---|---|---|
| `DEFAULT_CAPITAL_PKR` | 5,000 | 1,000–1M PKR | Starting capital |
| `RISK_PER_TRADE_PCT` | 0.02 | 0.01–0.05 | Risk per trade |
| `COMMISSION_PCT` | 0.001 | 0–0.005 | Per-side commission |
| `RSI_PERIOD` | 14 | 10–30 | RSI lookback |
| `RSI_OVERSOLD` | 40 | 30–50 | Oversold threshold |
| `RSI_OVERBOUGHT` | 65 | 60–80 | Overbought threshold |
| `MA_SHORT` | 20 | 10–30 | Fast MA period |
| `MA_LONG` | 50 | 30–200 | Slow MA period |
| `VOLUME_MA_PERIOD` | 20 | 10–50 | Volume MA period |
| `SWING_WINDOW` | 5 | 3–10 | Swing detection window |
