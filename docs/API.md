# API Reference

## Agent Package (`agent/`)

### `agent.__init__`

```python
from agent import (
    get_llm,           # LLM factory function
    build_agent,       # Single ReAct agent builder
    run_query,         # Synchronous query runner
    stream_query,      # Streaming query runner
    PSXTradingSupervisor,  # Multi-agent supervisor class
    build_technical_analyst,  # Technical specialist builder
    build_shariah_analyst,    # Shariah specialist builder
    build_risk_analyst,       # Risk specialist builder
)
```

### `agent.trading_agent`

#### `get_llm(provider: str | None = None, api_key: str | None = None) -> BaseChatModel`
Return a LangChain chat model for the given provider.

- **provider**: `"groq"` | `"gemini"` | `"openai_compatible"` (default from config)
- **api_key**: Override the env-var key

#### `build_agent(llm: BaseChatModel)`
Build a LangGraph ReAct agent with all trading tools (`ALL_TOOLS`).

#### `run_query(agent, user_message: str, chat_history: list[dict] | None = None) -> str`
Run a synchronous query through the agent.

#### `stream_query(agent, user_message: str, chat_history: list[dict] | None = None) -> Iterator[str]`
Stream the agent's response token by token. Yields text chunks.
Designed for Streamlit's `st.chat_message` streaming display.

### `agent.supervisor_agent`

#### `class PSXTradingSupervisor(llm: BaseChatModel)`
Multi-agent orchestrator with supervisor + 3 specialist agents.

**Methods:**
- `run(query: str) -> str` — Run query through multi-agent pipeline
- `stream(query: str) -> Iterator[str]` — Stream query results

The supervisor classifies queries by keyword and routes to:
1. `technical_analyst` — Price/Fib/RSI/Wave analysis
2. `shariah_analyst` — KMI compliance screening
3. `risk_analyst` — Backtesting/risk metrics

### `agent.specialized_agents`

#### `build_technical_analyst(llm: BaseChatModel)`
Build technical sub-agent (tools: `get_stock_summary`, `analyze_technical`).

#### `build_shariah_analyst(llm: BaseChatModel)`
Build Shariah sub-agent (tools: `check_shariah`, `list_shariah_watchlist`).

#### `build_risk_analyst(llm: BaseChatModel)`
Build risk sub-agent (tool: `run_backtest`).

---

## Tools Package (`tools/`)

### `tools.data_tools`

#### `fetch_ohlcv(ticker: str, days: int = 365) -> pd.DataFrame`
Fetch OHLCV data from PSX via `psxdata` package.

- **Returns**: DataFrame with columns `[Open, High, Low, Close, Volume]`

#### `get_current_price(ticker: str) -> float`
Return the latest closing price for a ticker.

#### `@tool get_stock_summary(query: str) -> str`
LangChain tool. Input format: `"TICKER"` or `"TICKER,DAYS"`.
Returns text summary with price, volume, trend, and data range.

### `tools.technical_tools`

#### `detect_swings(df: pd.DataFrame, window: int = 5) -> tuple[pd.Series, pd.Series]`
Find swing highs and lows using rolling window comparison.

#### `calculate_fib_levels(df, swing_low=None, swing_high=None) -> dict[str, float]`
Calculate Fibonacci retracement including bisection (50%) and trisection (33.3%/66.7%).

**Returns**: `{"23.6": ..., "33.3": ..., "38.2": ..., "50.0": ..., "61.8": ..., "66.7": ..., "78.6": ..., "swing_low": ..., "swing_high": ..., "diff": ...}`

#### `calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series`
Wilder's RSI calculation.

#### `add_indicators(df: pd.DataFrame) -> pd.DataFrame`
Add RSI, MA20, MA50, Volume MA, Vol Ratio columns.

#### `detect_wave_structure(df: pd.DataFrame) -> dict`
Simplified Elliott Wave trend detection.

**Returns**: `{"trend", "wave_position", "confidence", "description"}`

#### `generate_signal(df: pd.DataFrame) -> dict`
Full signal generation: Fib + RSI + Volume + Wave.

**Returns**: `{"signal", "reason", "close", "rsi", "vol_ratio", "fib_levels", "stop_loss", "take_profit", "wave", "ma_short", "ma_long", "ma_signal"}`

**Signal values**: `BUY` | `BUY_WEAK` | `SELL/TAKE_PROFIT` | `WATCH` | `WAIT`

#### `@tool analyze_technical(query: str) -> str`
LangChain tool. Runs full technical analysis. Input: `"TICKER"` or `"TICKER,DAYS"`.

### `tools.backtest_tools`

#### `class FibBacktestEngine(capital: float = 5000)`
Walk-forward Fibonacci backtesting engine.

**Methods:**
- `run(df: pd.DataFrame, ticker: str) -> dict` — Run backtest

#### `@tool run_backtest(query: str) -> str`
LangChain tool. Input: `"TICKER"` or `"TICKER,DAYS"` or `"TICKER,DAYS,CAPITAL"`.

#### `get_backtest_dataframes(ticker, days=365, capital=5000) -> tuple[dict, pd.DataFrame]`
Returns (metrics_dict, trades_df) for Streamlit chart rendering.

### `tools.shariah_tools`

#### `check_compliance(ticker: str) -> dict`
Core compliance check. Returns `{"ticker", "status", "reason", "in_kmi30", "note", "price_pkr", "low_price"}`.

#### `get_compliant_watchlist(max_price=10) -> list[dict]`
Return all compliant tickers with price ≤ max_price.

#### `@tool check_shariah(ticker: str) -> str`
LangChain tool. Check a ticker's Shariah compliance.

#### `@tool list_shariah_watchlist(_: str = "") -> str`
LangChain tool. List all compliant low-price stocks.

### `tools.chart_tools`

#### `create_price_chart(df, ticker, show_fib=True, show_ma=True, show_swings=True, buy_signals=None, sell_signals=None) -> go.Figure`
Professional candlestick chart with:
- Fibonacci overlay (colored, labeled levels)
- Moving averages (MA20/MA50)
- Swing high/low markers
- Buy/sell signal markers
- RSI subplot (with threshold lines)
- Volume subplot (green/red bars)

#### `create_equity_chart(equity_curve, trades, ticker, start_capital) -> go.Figure`
Backtest equity curve with trade markers.

#### `create_comparison_chart(tickers: list[str], days: int = 180) -> go.Figure`
Normalized multi-stock performance comparison (base=100).

---

## Configuration (`config.py`)

| Constant | Default | Description |
|---|---|---|
| `MODEL_PROVIDER` | `"groq"` | Default LLM provider |
| `PROVIDER_CONFIGS` | — | Dict of provider settings |
| `DEFAULT_CAPITAL_PKR` | `5_000` | Starting capital |
| `RISK_PER_TRADE_PCT` | `0.02` | 2% risk per trade |
| `MAX_OPEN_POSITIONS` | `3` | Max concurrent trades |
| `COMMISSION_PCT` | `0.001` | 0.1% per side |
| `DEFAULT_LOOKBACK_DAYS` | `365` | Default data window |
| `SWING_WINDOW` | `5` | Swing detection bars |
| `FIB_LEVELS` | — | All Fibonacci ratios |
| `BUY_ZONE_LOW` | `0.500` | 50% retracement |
| `BUY_ZONE_HIGH` | `0.618` | 61.8% retracement |
| `STOP_LOSS_FIB` | `0.786` | Stop at 78.6% |
| `TAKE_PROFIT_FIB` | `0.236` | TP at 23.6% |
| `RSI_PERIOD` | `14` | RSI lookback |
| `RSI_OVERSOLD` | `47` | Oversold threshold (PSX relaxed) |
| `RSI_OVERBOUGHT` | `65` | Overbought threshold |
| `VOLUME_RATIO_THRESHOLD` | `0.8` | Min volume ratio for entry |
| `MA_SHORT` | `20` | Fast MA |
| `MA_LONG` | `50` | Slow MA |
| `VOLUME_MA_PERIOD` | `20` | Volume MA |
| `SHARIAH_WATCHLIST` | — | Low-price compliant stocks |
| `KMI_30` | — | KMI-30 constituents |
| `LOW_PRICE_THRESHOLD_PKR` | `10` | Low price cutoff |
