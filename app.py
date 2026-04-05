"""
app.py — PSX Trading Decision Support System
Streamlit UI: Chat agent + Charts + Backtest panel
"""
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ── Page Config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="PSX Trading Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: #0d1117; }
  .main-title { font-size: 1.8rem; font-weight: 700; color: #26a69a; margin-bottom: 0; }
  .sub-title  { font-size: 0.9rem; color: #8b949e; margin-top: 0; }
  .signal-box { padding: 12px 16px; border-radius: 8px; margin: 8px 0; font-weight: 600; }
  .signal-buy    { background: #0d3322; border-left: 4px solid #26a69a; color: #26a69a; }
  .signal-sell   { background: #3b1111; border-left: 4px solid #ef5350; color: #ef5350; }
  .signal-watch  { background: #2d2400; border-left: 4px solid #f0c040; color: #f0c040; }
  .signal-wait   { background: #1c2128; border-left: 4px solid #8b949e; color: #8b949e; }
  .metric-card   { background: #161b22; border-radius: 8px; padding: 12px; text-align: center; }
  div[data-testid="stChatMessage"] { background: #161b22 !important; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Session State Defaults ────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "agent":         None,
        "chat_history":  [],
        "provider":      "groq",
        "api_key":       "",
        "agent_ready":   False,
        "last_ticker":   "WTL",
        "last_days":     365,
        "last_capital":  5000,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Sidebar — Configuration ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # LLM Provider
    provider = st.selectbox(
        "LLM Provider",
        ["groq", "gemini", "openai_compatible"],
        index=["groq", "gemini", "openai_compatible"].index(st.session_state.provider),
        help="Groq is the recommended free option (fastest, best tool-calling).",
    )
    st.session_state.provider = provider

    key_labels = {
        "groq":              "Groq API Key (console.groq.com)",
        "gemini":            "Google API Key (aistudio.google.com)",
        "openai_compatible": "OpenAI-compatible API Key",
    }
    api_key = st.text_input(
        key_labels[provider],
        type="password",
        value=st.session_state.api_key,
        placeholder="Paste your API key here…",
    )
    st.session_state.api_key = api_key

    if provider == "openai_compatible":
        base_url = st.text_input(
            "Base URL",
            value=os.getenv("OAI_BASE_URL", "https://api.openai.com/v1"),
        )
        os.environ["OAI_BASE_URL"] = base_url

    # Init Agent button
    if st.button("🚀 Connect Agent", use_container_width=True, type="primary"):
        if not api_key:
            st.error("Enter your API key first.")
        else:
            with st.spinner("Connecting to LLM…"):
                try:
                    from agent.trading_agent import get_llm, build_agent
                    llm   = get_llm(provider=provider, api_key=api_key)
                    agent = build_agent(llm)
                    st.session_state.agent       = agent
                    st.session_state.agent_ready = True
                    st.success("✅ Agent ready!")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

    # Status indicator
    if st.session_state.agent_ready:
        st.markdown("🟢 **Agent connected**")
    else:
        st.markdown("🔴 **Agent not connected**")

    st.divider()

    # Trading parameters
    st.markdown("## 📊 Chart & Backtest")
    selected_ticker = st.text_input(
        "Ticker Symbol", value=st.session_state.last_ticker,
        placeholder="e.g. WTL, KEL, UNITY"
    ).upper().strip()
    selected_days   = st.slider("Lookback (days)", 90, 730, st.session_state.last_days)
    capital         = st.number_input(
        "Capital (PKR)", min_value=1000, max_value=1_000_000,
        value=st.session_state.last_capital, step=1000,
    )
    st.session_state.last_ticker  = selected_ticker
    st.session_state.last_days    = selected_days
    st.session_state.last_capital = capital

    if st.button("📈 Load Chart", use_container_width=True):
        st.session_state["load_chart"] = True

    if st.button("🔁 Run Backtest", use_container_width=True):
        st.session_state["run_bt"] = True

    st.divider()

    # Quick access stocks
    st.markdown("## ⚡ Quick Select")
    watchlist = ["WTL", "CNERGY", "UNITY", "KEL", "LOTCHEM",
                 "MLCF", "SNGP", "KAPCO", "PAEL", "PIBTL"]
    cols = st.columns(2)
    for i, sym in enumerate(watchlist):
        if cols[i % 2].button(sym, key=f"qs_{sym}"):
            st.session_state.last_ticker = sym
            st.session_state["load_chart"] = True
            st.rerun()

    st.divider()
    st.markdown("*📌 Data via `psx` package · PSX historical data*")


# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📈 PSX Trading Analyst</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Fibonacci · Elliott Wave · Shariah-Compliant · Agentic AI</p>',
    unsafe_allow_html=True,
)

# Two-column layout: Chat (left) | Charts (right)
chat_col, chart_col = st.columns([1, 1], gap="large")

# ── LEFT: AI Agent Chat ───────────────────────────────────────────────────────
with chat_col:
    st.markdown("### 🤖 AI Trading Analyst")

    # Suggested prompts
    if not st.session_state.chat_history:
        st.markdown("**Try asking:**")
        suggestions = [
            f"Analyse {st.session_state.last_ticker} — should I buy?",
            f"Is {st.session_state.last_ticker} Shariah-compliant?",
            f"Backtest the strategy on {st.session_state.last_ticker}",
            "List all Shariah-compliant low-price stocks",
            f"What are the Fibonacci levels for {st.session_state.last_ticker}?",
        ]
        sug_cols = st.columns(2)
        for i, sug in enumerate(suggestions):
            if sug_cols[i % 2].button(sug, key=f"sug_{i}"):
                st.session_state["pending_input"] = sug
                st.rerun()

    # Chat history display
    chat_container = st.container(height=420)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])

    # Input box
    user_input = st.chat_input(
        "Ask about a stock, Fibonacci levels, signals, backtest…",
        disabled=not st.session_state.agent_ready,
    )

    # Handle pending input from suggestion buttons
    if "pending_input" in st.session_state and st.session_state.pending_input:
        user_input = st.session_state.pop("pending_input")

    if user_input:
        if not st.session_state.agent_ready:
            st.warning("Connect the agent first (sidebar → Connect Agent).")
        else:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            with chat_container:
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(user_input)

                with st.chat_message("assistant", avatar="🤖"):
                    response_box = st.empty()
                    full_response = ""

                    from agent.trading_agent import stream_query
                    with st.spinner("Thinking…"):
                        try:
                            for chunk in stream_query(
                                st.session_state.agent,
                                user_input,
                                st.session_state.chat_history[:-1],
                            ):
                                full_response = chunk
                                response_box.markdown(full_response + " ▌")
                            response_box.markdown(full_response)
                        except Exception as e:
                            full_response = f"⚠️ Agent error: {e}"
                            response_box.markdown(full_response)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": full_response}
            )

    if st.button("🗑️ Clear chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()


# ── RIGHT: Charts & Backtest ──────────────────────────────────────────────────
with chart_col:
    ticker = st.session_state.last_ticker
    days   = st.session_state.last_days

    tab_chart, tab_backtest, tab_compare = st.tabs(["📊 Chart", "📉 Backtest", "🔁 Compare"])

    # ── Tab 1: Price Chart ────────────────────────────────────────────────
    with tab_chart:
        should_load = st.session_state.pop("load_chart", False)

        if should_load or ticker:
            with st.spinner(f"Loading {ticker}…"):
                try:
                    from tools.data_tools import fetch_ohlcv
                    from tools.technical_tools import (
                        add_indicators, generate_signal, calculate_fib_levels,
                    )
                    from tools.chart_tools import create_price_chart

                    df  = fetch_ohlcv(ticker, days=days)
                    sig = generate_signal(df)
                    fig = create_price_chart(df, ticker)

                    # Signal banner
                    signal = sig["signal"]
                    signal_class = {
                        "BUY":             "signal-buy",
                        "BUY_WEAK":        "signal-buy",
                        "SELL/TAKE_PROFIT":"signal-sell",
                        "WATCH":           "signal-watch",
                        "WAIT":            "signal-wait",
                    }.get(signal, "signal-wait")

                    st.markdown(
                        f'<div class="signal-box {signal_class}">'
                        f'Signal: {signal} &nbsp;|&nbsp; '
                        f'RSI: {sig["rsi"]} &nbsp;|&nbsp; '
                        f'Close: {sig["close"]} PKR &nbsp;|&nbsp; '
                        f'Vol: {sig["vol_ratio"]}x'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Quick metrics
                    m1, m2, m3, m4 = st.columns(4)
                    fib = sig["fib_levels"]
                    m1.metric("Buy Zone Low",  f"{fib.get('61.8', 0):.3f}")
                    m2.metric("Buy Zone High", f"{fib.get('50.0', 0):.3f}")
                    m3.metric("Stop Loss",     f"{sig['stop_loss']:.3f}")
                    m4.metric("Take Profit",   f"{sig['take_profit']:.3f}")

                    st.plotly_chart(fig, use_container_width=True)

                    # Wave info
                    wave = sig["wave"]
                    st.caption(
                        f"🌊 **Elliott Wave**: {wave['trend']} — {wave['description']}"
                    )

                except Exception as e:
                    st.error(f"Chart error: {e}")
        else:
            st.info("Select a ticker in the sidebar and click **Load Chart**.")

    # ── Tab 2: Backtest ───────────────────────────────────────────────────
    with tab_backtest:
        should_bt = st.session_state.pop("run_bt", False)

        if should_bt:
            with st.spinner(f"Backtesting {ticker} over {days} days…"):
                try:
                    from tools.backtest_tools import get_backtest_dataframes
                    from tools.chart_tools    import create_equity_chart

                    metrics, trades_df = get_backtest_dataframes(
                        ticker, days=days, capital=float(capital)
                    )

                    if "error" in metrics:
                        st.warning(metrics["error"])
                    else:
                        # Performance metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric(
                            "Total Return",
                            f"{metrics['total_return_pct']:+.1f}%",
                            f"vs B&H {metrics['buy_hold_pct']:+.1f}%",
                        )
                        col2.metric("Win Rate",    f"{metrics['win_rate_pct']}%")
                        col3.metric("Max Drawdown",f"{metrics['max_drawdown_pct']:.1f}%")
                        col4.metric("Sharpe",      f"{metrics['sharpe_ratio']}")

                        col5, col6, col7, col8 = st.columns(4)
                        col5.metric("Profit Factor", metrics["profit_factor"])
                        col6.metric("Total Trades",  metrics["total_trades"])
                        col7.metric(
                            "Net P&L",
                            f"{metrics['net_pnl_pkr']:+.0f} PKR",
                        )
                        col8.metric(
                            "End Capital",
                            f"{metrics['end_capital']:.0f} PKR",
                            f"Start: {metrics['start_capital']:.0f}",
                        )

                        # Equity curve
                        eq_fig = create_equity_chart(
                            metrics["equity_curve"],
                            metrics["trades"],
                            ticker,
                            metrics["start_capital"],
                        )
                        st.plotly_chart(eq_fig, use_container_width=True)

                        # Trade log table
                        if not trades_df.empty:
                            st.markdown("#### Trade Log")
                            display_cols = [
                                "entry_date", "exit_date", "entry_price",
                                "exit_price", "shares", "pnl_pkr", "pnl_pct",
                                "exit_reason", "won",
                            ]
                            display_df = trades_df[
                                [c for c in display_cols if c in trades_df.columns]
                            ].copy()

                            # Colour rows
                            def style_row(row):
                                color = "#0d3322" if row.get("won", False) else "#3b1111"
                                return [f"background-color: {color}"] * len(row)

                            st.dataframe(
                                display_df.style.apply(style_row, axis=1),
                                use_container_width=True,
                                height=250,
                            )

                except Exception as e:
                    st.error(f"Backtest error: {e}")
        else:
            st.info("Set parameters in the sidebar and click **Run Backtest**.")

    # ── Tab 3: Multi-Stock Compare ────────────────────────────────────────
    with tab_compare:
        compare_tickers_raw = st.text_input(
            "Tickers to compare (comma-separated)",
            value="WTL,CNERGY,KEL,UNITY",
            placeholder="WTL,KEL,SNGP,UNITY",
        )
        compare_days = st.slider("Comparison period (days)", 90, 365, 180, key="cmp_days")

        if st.button("📊 Compare", use_container_width=True):
            tickers_list = [t.strip().upper() for t in compare_tickers_raw.split(",") if t.strip()]
            with st.spinner("Fetching data…"):
                try:
                    from tools.chart_tools import create_comparison_chart
                    cmp_fig = create_comparison_chart(tickers_list, compare_days)
                    st.plotly_chart(cmp_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Comparison error: {e}")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer:** This tool is for educational and decision-support purposes only. "
    "It does not constitute financial advice. Always do your own research before investing. "
    "Past backtest results do not guarantee future returns. "
    "Verify Shariah compliance with certified Islamic finance scholars / PSX KMI index."
)
