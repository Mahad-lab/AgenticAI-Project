"""
tools/backtest_tools.py — Strategy backtesting engine
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from config import (
    DEFAULT_CAPITAL_PKR, RISK_PER_TRADE_PCT, COMMISSION_PCT,
    BUY_ZONE_LOW, BUY_ZONE_HIGH, STOP_LOSS_FIB, TAKE_PROFIT_FIB,
    RSI_OVERSOLD, RSI_OVERBOUGHT, SWING_WINDOW,
)
from tools.data_tools import fetch_ohlcv
from tools.technical_tools import (
    add_indicators, calculate_fib_levels, last_significant_swing,
)


# ─── Trade Record ─────────────────────────────────────────────────────────────

@dataclass
class Trade:
    ticker:      str
    entry_date:  str
    exit_date:   str
    entry_price: float
    exit_price:  float
    shares:      int
    capital_used:float
    exit_reason: str  # "take_profit" | "stop_loss" | "end_of_data"
    pnl_pkr:     float = field(init=False)
    pnl_pct:     float = field(init=False)
    won:         bool  = field(init=False)

    def __post_init__(self):
        commission   = (self.entry_price + self.exit_price) * self.shares * COMMISSION_PCT
        self.pnl_pkr = (self.exit_price - self.entry_price) * self.shares - commission
        self.pnl_pct = self.pnl_pkr / self.capital_used * 100
        self.won     = self.pnl_pkr > 0


# ─── Core Backtesting Engine ─────────────────────────────────────────────────

class FibBacktestEngine:
    """
    Fibonacci retracement strategy backtester.

    Entry rule   : Price retraces to 50–61.8% Fib level AND RSI < threshold
                   AND volume is above 20-day average
    Stop loss    : Price falls below 78.6% Fib level
    Take profit  : Price recovers to 23.6% Fib level
    Position size: Risk-based — never risk more than RISK_PER_TRADE_PCT of capital
    """

    def __init__(self, capital: float = DEFAULT_CAPITAL_PKR):
        self.capital       = capital
        self.equity_curve  = []
        self.trades: List[Trade] = []
        self._current_capital    = capital

    def _position_size(self, entry: float, stop: float) -> int:
        """
        Calculate shares to buy using fixed-fractional risk.
        Risk = RISK_PER_TRADE_PCT × capital
        Shares = risk / (entry - stop)
        """
        risk_amount = self._current_capital * RISK_PER_TRADE_PCT
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0:
            return 0
        shares = math.floor(risk_amount / risk_per_share)
        # Never use more than 50% of current capital in one trade
        max_shares = math.floor(self._current_capital * 0.5 / entry)
        return max(1, min(shares, max_shares))

    def run(self, df: pd.DataFrame, ticker: str) -> dict:
        """
        Walk-forward backtest over the full DataFrame.
        Uses a rolling 90-bar window to recalculate Fib levels.
        """
        df_ind = add_indicators(df)
        self.equity_curve = [self._current_capital]
        self.trades       = []

        in_trade   = False
        entry_bar  = None
        entry_price = stop_loss = take_profit = 0.0
        shares     = 0
        fib_window = 90   # recalculate Fib every N bars or on new entry

        for i in range(60, len(df_ind)):            # warm-up: 60 bars
            bar   = df_ind.iloc[i]
            close = bar["Close"]
            rsi   = bar["RSI"]
            vol_r = bar["Vol_Ratio"]

            # ── Exit logic ────────────────────────────────────────────────
            if in_trade:
                exit_reason = None
                exit_price  = close

                if close >= take_profit:
                    exit_reason = "take_profit"
                elif close <= stop_loss:
                    exit_reason = "stop_loss"
                elif i == len(df_ind) - 1:
                    exit_reason = "end_of_data"

                if exit_reason:
                    trade = Trade(
                        ticker      = ticker,
                        entry_date  = str(df_ind.index[entry_bar].date()),
                        exit_date   = str(df_ind.index[i].date()),
                        entry_price = entry_price,
                        exit_price  = exit_price,
                        shares      = shares,
                        capital_used= entry_price * shares,
                        exit_reason = exit_reason,
                    )
                    self.trades.append(trade)
                    self._current_capital += trade.pnl_pkr
                    in_trade = False

            # ── Entry logic ───────────────────────────────────────────────
            if not in_trade and not (pd.isna(rsi) or pd.isna(vol_r)):
                window_df = df_ind.iloc[max(0, i - fib_window): i + 1]
                fib       = calculate_fib_levels(window_df)

                buy_low  = fib["61.8"]
                buy_high = fib["50.0"]
                sl       = fib["78.6"]
                tp       = fib["23.6"]

                if (buy_low <= close <= buy_high
                        and rsi < RSI_OVERSOLD
                        and vol_r >= 1.0
                        and close > sl          # sanity: SL is below entry
                        and tp > close):        # sanity: TP is above entry

                    shares = self._position_size(close, sl)
                    if shares > 0 and close * shares <= self._current_capital:
                        in_trade    = True
                        entry_bar   = i
                        entry_price = close
                        stop_loss   = sl
                        take_profit = tp

            self.equity_curve.append(round(self._current_capital, 2))

        return self._compute_metrics(df_ind)

    def _compute_metrics(self, df: pd.DataFrame) -> dict:
        trades   = self.trades
        equity   = pd.Series(self.equity_curve)

        if not trades:
            return {"error": "No trades were triggered. Try a longer data window "
                             "or a stock with more volatility."}

        total_trades   = len(trades)
        wins           = sum(1 for t in trades if t.won)
        losses         = total_trades - wins
        win_rate       = wins / total_trades * 100

        gross_profit   = sum(t.pnl_pkr for t in trades if t.won)
        gross_loss     = sum(t.pnl_pkr for t in trades if not t.won)
        net_pnl        = sum(t.pnl_pkr for t in trades)
        profit_factor  = (abs(gross_profit / gross_loss)
                          if gross_loss != 0 else float("inf"))

        avg_win  = gross_profit / wins   if wins   else 0
        avg_loss = gross_loss   / losses if losses else 0

        # Max drawdown on equity curve
        roll_max  = equity.cummax()
        drawdown  = (equity - roll_max) / roll_max * 100
        max_dd    = drawdown.min()

        # Buy-and-hold comparison
        bh_return = ((df["Close"].iloc[-1] - df["Close"].iloc[60])
                     / df["Close"].iloc[60] * 100)

        start_cap = DEFAULT_CAPITAL_PKR
        end_cap   = self._current_capital
        total_ret = (end_cap - start_cap) / start_cap * 100

        # Sharpe ratio (annualised, assume 250 trading days)
        daily_rets   = equity.pct_change().dropna()
        sharpe       = (daily_rets.mean() / daily_rets.std() * np.sqrt(250)
                        if daily_rets.std() > 0 else 0)

        # Build trade log
        trade_log = [
            {
                "entry_date":  t.entry_date,
                "exit_date":   t.exit_date,
                "entry_price": round(t.entry_price, 2),
                "exit_price":  round(t.exit_price,  2),
                "shares":      t.shares,
                "pnl_pkr":     round(t.pnl_pkr, 2),
                "pnl_pct":     round(t.pnl_pct, 2),
                "exit_reason": t.exit_reason,
                "won":         t.won,
            }
            for t in trades
        ]

        return {
            "ticker":           trades[0].ticker if trades else "N/A",
            "start_capital":    round(start_cap, 2),
            "end_capital":      round(end_cap,   2),
            "net_pnl_pkr":      round(net_pnl,   2),
            "total_return_pct": round(total_ret,  2),
            "buy_hold_pct":     round(bh_return,  2),
            "total_trades":     total_trades,
            "win_rate_pct":     round(win_rate,   2),
            "profit_factor":    round(profit_factor, 2),
            "avg_win_pkr":      round(avg_win,  2),
            "avg_loss_pkr":     round(avg_loss, 2),
            "max_drawdown_pct": round(max_dd,   2),
            "sharpe_ratio":     round(sharpe,   2),
            "equity_curve":     equity.tolist(),
            "trades":           trade_log,
        }


# ─── LangChain Tool ───────────────────────────────────────────────────────────

@tool
def run_backtest(query: str) -> str:
    """
    Backtest the Fibonacci retracement strategy on a PSX stock.

    Input (plain string):  "TICKER"  or  "TICKER,DAYS"  or  "TICKER,DAYS,CAPITAL"
    Examples:
        "WTL"
        "KEL,365"
        "UNITY,730,10000"

    Returns a backtest performance report including:
      - Total return vs buy-and-hold
      - Win rate, profit factor, Sharpe ratio
      - Max drawdown
      - Individual trade log summary
    """
    parts   = [p.strip().upper() for p in query.split(",")]
    ticker  = parts[0]
    days    = int(parts[1])   if len(parts) > 1 else 365
    capital = float(parts[2]) if len(parts) > 2 else DEFAULT_CAPITAL_PKR

    try:
        df = fetch_ohlcv(ticker, days=days)
    except Exception as exc:
        return f"Data error: {exc}"

    if len(df) < 70:
        return (f"Not enough data for {ticker} ({len(df)} bars). "
                "Need at least 70 trading days.")

    engine  = FibBacktestEngine(capital=capital)
    metrics = engine.run(df, ticker)

    if "error" in metrics:
        return f"Backtest result: {metrics['error']}"

    trades = metrics["trades"]
    recent = trades[-5:] if len(trades) >= 5 else trades

    lines = [
        f"📈  Backtest Report — {ticker}  ({days}d, capital={capital:.0f} PKR)",
        "",
        "── Performance Summary ──",
        f"  Start Capital    : {metrics['start_capital']:>10.2f} PKR",
        f"  End Capital      : {metrics['end_capital']:>10.2f} PKR",
        f"  Net P&L          : {metrics['net_pnl_pkr']:>+10.2f} PKR",
        f"  Total Return     : {metrics['total_return_pct']:>+9.2f}%",
        f"  Buy-and-Hold     : {metrics['buy_hold_pct']:>+9.2f}%",
        f"  Alpha            : {metrics['total_return_pct'] - metrics['buy_hold_pct']:>+9.2f}%",
        "",
        "── Risk Metrics ──",
        f"  Total Trades     : {metrics['total_trades']}",
        f"  Win Rate         : {metrics['win_rate_pct']}%",
        f"  Profit Factor    : {metrics['profit_factor']}  (>1.5 is good)",
        f"  Avg Win          : {metrics['avg_win_pkr']:>+8.2f} PKR",
        f"  Avg Loss         : {metrics['avg_loss_pkr']:>+8.2f} PKR",
        f"  Max Drawdown     : {metrics['max_drawdown_pct']:.2f}%",
        f"  Sharpe Ratio     : {metrics['sharpe_ratio']}  (>1.0 is acceptable)",
        "",
        f"── Last {len(recent)} Trades ──",
    ]
    for t in recent:
        result_icon = "✅" if t["won"] else "❌"
        lines.append(
            f"  {result_icon} {t['entry_date']} → {t['exit_date']} | "
            f"Entry {t['entry_price']} → Exit {t['exit_price']} | "
            f"P&L {t['pnl_pkr']:+.2f} PKR ({t['pnl_pct']:+.1f}%) | {t['exit_reason']}"
        )

    return "\n".join(lines)


def get_backtest_dataframes(ticker: str, days: int = 365,
                            capital: float = DEFAULT_CAPITAL_PKR
                            ) -> tuple[dict, pd.DataFrame]:
    """
    Run backtest and return (metrics_dict, trades_dataframe).
    Used by Streamlit for chart rendering.
    """
    df      = fetch_ohlcv(ticker, days=days)
    engine  = FibBacktestEngine(capital=capital)
    metrics = engine.run(df, ticker)
    trades_df = pd.DataFrame(metrics.get("trades", []))
    return metrics, trades_df
