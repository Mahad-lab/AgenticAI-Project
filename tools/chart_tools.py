"""
tools/chart_tools.py — Plotly charts: candlestick + Fib levels + signals + equity curve
"""
from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from config import (
    MA_SHORT, MA_LONG, FIB_LEVELS, RSI_PERIOD,
)
from tools.data_tools import fetch_ohlcv
from tools.technical_tools import (
    add_indicators, calculate_fib_levels, last_significant_swing,
    detect_swings,
)

# ─── Colour Palette ───────────────────────────────────────────────────────────
COLOURS = {
    "green":    "#26a69a",
    "red":      "#ef5350",
    "bg":       "#0d1117",
    "grid":     "#1c2128",
    "text":     "#e6edf3",
    "fib_050":  "#f0c040",   # Bisection — yellow
    "fib_618":  "#4caf50",   # Key buy zone — green
    "fib_382":  "#42a5f5",   # 38.2% — blue
    "fib_236":  "#ab47bc",   # Take profit — purple
    "fib_786":  "#ef5350",   # Stop loss zone — red
    "fib_333":  "#80cbc4",   # Trisection — teal
    "fib_667":  "#80cbc4",
    "ma_short": "#ff9800",
    "ma_long":  "#29b6f6",
    "rsi":      "#ce93d8",
    "volume":   "#546e7a",
    "vol_high": "#4caf50",
    "signal_buy":  "#00e676",
    "signal_sell": "#ff5252",
}

FIB_COLOUR_MAP = {
    "23.6":  COLOURS["fib_236"],
    "33.3":  COLOURS["fib_333"],
    "38.2":  COLOURS["fib_382"],
    "50.0":  COLOURS["fib_050"],
    "61.8":  COLOURS["fib_618"],
    "66.7":  COLOURS["fib_667"],
    "78.6":  COLOURS["fib_786"],
}

DARK_LAYOUT = dict(
    paper_bgcolor=COLOURS["bg"],
    plot_bgcolor=COLOURS["bg"],
    font=dict(color=COLOURS["text"], family="Inter, sans-serif", size=12),
    xaxis=dict(showgrid=True, gridcolor=COLOURS["grid"], zeroline=False,
               rangeslider=dict(visible=False)),
    yaxis=dict(showgrid=True, gridcolor=COLOURS["grid"], zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    margin=dict(l=60, r=20, t=50, b=30),
)


# ─── Main Price Chart ─────────────────────────────────────────────────────────

def create_price_chart(
    df: pd.DataFrame,
    ticker: str,
    show_fib: bool      = True,
    show_ma: bool       = True,
    show_swings: bool   = True,
    buy_signals: pd.Series | None = None,
    sell_signals: pd.Series | None = None,
) -> go.Figure:
    """
    Candlestick chart with optional overlays:
      - Fibonacci retracement levels (with bisection / trisection labels)
      - Moving averages (MA20, MA50)
      - Swing high/low markers
      - Buy / sell signal markers
      - RSI subplot
      - Volume subplot
    """
    df = add_indicators(df)

    # 3-row subplot: [Price + overlays] [RSI] [Volume]
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.02,
    )

    # ── Candlestick ──────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color=COLOURS["green"],
        decreasing_line_color=COLOURS["red"],
        name=ticker,
    ), row=1, col=1)

    # ── Moving Averages ───────────────────────────────────────────────────
    if show_ma:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MA_short"], name=f"MA{MA_SHORT}",
            line=dict(color=COLOURS["ma_short"], width=1.5),
        ), row=1, col=1)
        if "MA_long" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["MA_long"], name=f"MA{MA_LONG}",
                line=dict(color=COLOURS["ma_long"], width=1.5),
            ), row=1, col=1)

    # ── Fibonacci Levels ──────────────────────────────────────────────────
    if show_fib:
        fib = calculate_fib_levels(df)
        x0, x1 = df.index[0], df.index[-1]

        for label, price in fib.items():
            if label in FIB_COLOUR_MAP:
                colour = FIB_COLOUR_MAP[label]
                dash   = "dot" if label in ("33.3", "66.7") else "dash"
                width  = 2 if label in ("50.0", "61.8") else 1
                suffix = ""
                if label == "50.0":  suffix = " ← bisect"
                if label in ("33.3", "66.7"): suffix = " ← trisect"
                if label == "61.8":  suffix = " ← buy zone"
                if label == "78.6":  suffix = " ← stop loss"
                if label == "23.6":  suffix = " ← take profit"

                fig.add_shape(
                    type="line", x0=x0, x1=x1,
                    y0=price, y1=price,
                    line=dict(color=colour, width=width, dash=dash),
                    row=1, col=1,
                )
                fig.add_annotation(
                    x=x1, y=price, xanchor="left",
                    text=f" {label}%  {price:.3f}{suffix}",
                    showarrow=False,
                    font=dict(color=colour, size=10),
                    row=1, col=1,
                )

    # ── Swing Highs / Lows ────────────────────────────────────────────────
    if show_swings:
        sh_mask, sl_mask = detect_swings(df)
        fig.add_trace(go.Scatter(
            x=df.index[sh_mask], y=df["High"][sh_mask] * 1.002,
            mode="markers", name="Swing High",
            marker=dict(symbol="triangle-down", color=COLOURS["red"], size=8),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index[sl_mask], y=df["Low"][sl_mask] * 0.998,
            mode="markers", name="Swing Low",
            marker=dict(symbol="triangle-up", color=COLOURS["green"], size=8),
        ), row=1, col=1)

    # ── Trade Signals ────────────────────────────────────────────────────
    if buy_signals is not None and not buy_signals.empty:
        sig_df = df[buy_signals]
        fig.add_trace(go.Scatter(
            x=sig_df.index, y=sig_df["Low"] * 0.995,
            mode="markers+text", text=["BUY"] * len(sig_df),
            textposition="bottom center", name="Buy Signal",
            marker=dict(symbol="triangle-up", color=COLOURS["signal_buy"], size=12),
        ), row=1, col=1)

    if sell_signals is not None and not sell_signals.empty:
        sig_df = df[sell_signals]
        fig.add_trace(go.Scatter(
            x=sig_df.index, y=sig_df["High"] * 1.005,
            mode="markers+text", text=["SELL"] * len(sig_df),
            textposition="top center", name="Sell Signal",
            marker=dict(symbol="triangle-down", color=COLOURS["signal_sell"], size=12),
        ), row=1, col=1)

    # ── RSI ──────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"], name=f"RSI({RSI_PERIOD})",
        line=dict(color=COLOURS["rsi"], width=1.5),
    ), row=2, col=1)
    for level, colour in [(70, COLOURS["red"]), (30, COLOURS["green"]), (50, "#555")]:
        fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1],
                      y0=level, y1=level,
                      line=dict(color=colour, width=1, dash="dot"),
                      row=2, col=1)

    # ── Volume ────────────────────────────────────────────────────────────
    vol_colours = [
        COLOURS["vol_high"] if df["Volume"].iloc[i] >= df["Vol_MA"].iloc[i]
        else COLOURS["volume"]
        for i in range(len(df))
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume", marker_color=vol_colours,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Vol_MA"], name="Vol MA",
        line=dict(color="#ff9800", width=1, dash="dot"),
    ), row=3, col=1)

    # ── Layout ────────────────────────────────────────────────────────────
    layout = {**DARK_LAYOUT,
               "title": dict(text=f"  {ticker} — Technical Analysis",
                             font=dict(size=18)),
               "height": 700,
               "yaxis":  dict(**DARK_LAYOUT.get("yaxis", {}),
                               title="Price (PKR)"),
               "yaxis2": dict(showgrid=True, gridcolor=COLOURS["grid"],
                               title="RSI", range=[0, 100]),
               "yaxis3": dict(showgrid=True, gridcolor=COLOURS["grid"],
                               title="Volume"),
               "xaxis3": dict(**DARK_LAYOUT.get("xaxis", {}),
                               title="Date"),
               }
    fig.update_layout(**layout)
    return fig


# ─── Equity Curve Chart ───────────────────────────────────────────────────────

def create_equity_chart(equity_curve: list[float],
                        trades: list[dict],
                        ticker: str,
                        start_capital: float) -> go.Figure:
    """Bar chart equity curve with trade markers."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=equity_curve, mode="lines",
        name="Strategy Equity",
        line=dict(color=COLOURS["green"], width=2),
        fill="tozeroy", fillcolor="rgba(38,166,154,0.1)",
    ))
    fig.add_hline(y=start_capital, line_dash="dot",
                  line_color="#555", annotation_text="Start Capital")

    # Mark trade exits on equity curve
    if trades:
        win_y  = [equity_curve[min(i * (len(equity_curve) // max(len(trades), 1))
                                    + (len(equity_curve) // max(len(trades), 1)),
                                    len(equity_curve) - 1)]
                   for i, t in enumerate(trades) if t["won"]]
        loss_y = [equity_curve[min(i * (len(equity_curve) // max(len(trades), 1))
                                    + (len(equity_curve) // max(len(trades), 1)),
                                    len(equity_curve) - 1)]
                   for i, t in enumerate(trades) if not t["won"]]

    layout = {**DARK_LAYOUT,
               "title": dict(text=f"  {ticker} — Equity Curve",
                             font=dict(size=16)),
               "height": 350,
               "yaxis": dict(**DARK_LAYOUT.get("yaxis", {}),
                              title="Capital (PKR)"),
               "xaxis": dict(**DARK_LAYOUT.get("xaxis", {}),
                              title="Bar #"),
               }
    fig.update_layout(**layout)
    return fig


# ─── Multi-Stock Comparison ───────────────────────────────────────────────────

def create_comparison_chart(tickers: list[str], days: int = 180) -> go.Figure:
    """Normalised price performance comparison (base = 100)."""
    fig = go.Figure()
    palette = ["#26a69a", "#ef5350", "#ff9800", "#42a5f5",
               "#ab47bc", "#ec407a", "#66bb6a", "#ffca28"]

    for i, ticker in enumerate(tickers[:8]):
        try:
            df = fetch_ohlcv(ticker, days=days)
            normalised = df["Close"] / df["Close"].iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=df.index, y=normalised, name=ticker,
                line=dict(color=palette[i % len(palette)], width=2),
            ))
        except Exception:
            continue

    layout = {**DARK_LAYOUT,
               "title": dict(text="  Relative Performance (base=100)",
                             font=dict(size=16)),
               "height": 400,
               "yaxis": dict(**DARK_LAYOUT.get("yaxis", {}),
                              title="Normalised Price"),
               "xaxis": dict(**DARK_LAYOUT.get("xaxis", {}),
                              title="Date"),
               }
    fig.update_layout(**layout)
    return fig
