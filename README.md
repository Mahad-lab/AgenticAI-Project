# PSX Trading Decision Support System

AI-powered technical analysis tool for the Pakistan Stock Exchange (PSX).  
Built with LangGraph ReAct agents, Fibonacci analysis, and Streamlit.

---

## Features

- **Fibonacci Retracement** — 23.6%, 38.2%, 50%, 61.8%, 78.6% levels
- **Bisection & Trisection** — 50%, 33.3%, 66.7% price levels
- **Simplified Elliott Wave** — trend direction and wave position detection
- **RSI + Volume confirmation** — filters for higher-quality signals
- **Strategy Backtester** — walk-forward simulation with position sizing
- **Shariah Compliance Filter** — KMI-30 / KMI All-Share screening
- **Multi-Provider AI Agent** — Groq / Gemini / OpenAI-compatible
- **Streamlit Dashboard** — chat interface + interactive charts

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and add your key:
#   GROQ_API_KEY=gsk_...        ← get free key at console.groq.com
#   MODEL_PROVIDER=groq
```

### 3. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Getting a Free Groq API Key (Recommended)

1. Go to https://console.groq.com
2. Sign up (free, no credit card needed)
3. Create an API key
4. Paste it in the Streamlit sidebar or `.env` file

Groq gives you **14,400 free requests/day** using `llama-3.3-70b-versatile`.

---

## Alternatively: Google Gemini (Free)

1. Go to https://aistudio.google.com
2. Get a free API key
3. Set `MODEL_PROVIDER=gemini` and `GOOGLE_API_KEY=...` in `.env`

---

## Project Structure

```
psx_trader/
├── app.py                    ← Streamlit UI (entry point)
├── config.py                 ← All settings (tickers, capital, Fib params)
├── requirements.txt
├── .env.example
│
├── tools/
│   ├── data_tools.py         ← psx-feed wrapper + LangChain tool
│   ├── technical_tools.py    ← Fibonacci, RSI, Elliott Wave, signals
│   ├── backtest_tools.py     ← Walk-forward backtesting engine
│   ├── chart_tools.py        ← Plotly candlestick + Fib charts
│   └── shariah_tools.py      ← KMI compliance screening
│
└── agent/
    ├── trading_agent.py      ← LangGraph ReAct agent + LLM factory
    └── prompts.py            ← System prompt for the analyst persona
```

---

## Usage in Jupyter Notebook

The core logic is fully importable without Streamlit:

```python
from tools.data_tools      import fetch_ohlcv
from tools.technical_tools import generate_signal, calculate_fib_levels
from tools.backtest_tools  import FibBacktestEngine
from tools.chart_tools     import create_price_chart

# Fetch data
df = fetch_ohlcv("WTL", days=365)

# Get signal
sig = generate_signal(df)
print(sig["signal"], sig["reason"])

# Fibonacci levels
fib = calculate_fib_levels(df)
print(fib)

# Run backtest
engine  = FibBacktestEngine(capital=5000)
metrics = engine.run(df, "WTL")
print(f"Return: {metrics['total_return_pct']}%  |  Win rate: {metrics['win_rate_pct']}%")

# Chart (in notebook)
fig = create_price_chart(df, "WTL")
fig.show()
```

---

## Strategy Logic

**Entry (BUY):**
- Price retraces to the 50–61.8% Fibonacci zone
- RSI < 40 (oversold)
- Volume ≥ 1× the 20-day average

**Exit (TAKE PROFIT):**
- Price recovers to the 23.6% Fibonacci level

**Exit (STOP LOSS):**
- Price falls below the 78.6% Fibonacci level

**Position Sizing:**
- Risk = 2% of current capital per trade
- Shares = risk_amount / (entry_price - stop_loss_price)
- Never exceed 50% of capital in one trade

---

## Shariah Compliance

All signals respect the KMI All-Share screening.  
Conventional banks (MCB, HBL, UBL, etc.) are excluded.  
Always verify the latest KMI list at: https://www.psx.com.pk → Indices → KMI

---

## Disclaimer

This tool is for educational and decision-support purposes only.  
It does not constitute financial advice.  
Always do your own research before investing.  
Past backtest results do not guarantee future returns.
