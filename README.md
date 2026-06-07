<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-1.38+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/LangGraph-0.2+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/PSX-Data-00B4D8?style=for-the-badge&logo=chart&logoColor=white" alt="PSX"/>
  <img src="https://img.shields.io/badge/Multi--Agent-AI-8B5CF6?style=for-the-badge&logo=robot&logoColor=white" alt="Multi-Agent"/>
</p>

# 📈 PSX Multi-Agent Trading Decision Support System

**AI-powered technical analysis platform for the Pakistan Stock Exchange (PSX).**
Built with a collaborative multi-agent AI system, Fibonacci analysis, Shariah compliance screening, and an interactive Streamlit dashboard.

<p align="center">
  <b>🎯 Purpose:</b> Empower small retail investors (5,000–20,000 PKR) with professional-grade,
  AI-driven trading analysis — combining technical indicators, risk management,
  and Islamic finance compliance.
</p>

---

## 🌟 Key Features

| Feature | Description |
|---|---|
| **🤖 Multi-Agent AI System** | Supervisor delegates to 3 specialist agents: Technical, Shariah, Risk |
| **📐 Fibonacci Analysis** | 23.6%–78.6% retracement levels + bisection (50%) & trisection (33.3%/66.7%) |
| **🌊 Elliott Wave Detection** | Simplified impulse/corrective wave structure identification |
| **📊 RSI + Volume Confirmation** | Multi-filter signal generation (oversold/overbought + volume spike) |
| **🕌 Shariah Compliance** | KMI-30 / KMI All-Share screening with compliance notes |
| **🔁 Strategy Backtester** | Walk-forward simulation with risk-based position sizing |
| **📈 Interactive Charts** | Professional Plotly candlestick charts with Fib overlays |
| **☁️ Multi-Provider LLM** | Groq (free), Google Gemini (free), OpenAI-compatible |
| **🔬 Single Agent Mode** | Classic ReAct agent for simpler queries |

---

## 🏗️ Multi-Agent Architecture

```
                    ┌──────────────────────────────────────┐
                    │        User Query (Streamlit UI)      │
                    └────────────────┬─────────────────────┘
                                     │
                    ┌────────────────▼─────────────────────┐
                    │         SUPERVISOR AGENT              │
                    │  Intent Classification + Routing      │
                    └──────┬──────────┬──────────┬─────────┘
                           │          │          │
              ┌────────────▼──┐ ┌─────▼──────┐ ┌▼─────────────┐
              │  TECHNICAL    │ │  SHARIAH   │ │    RISK      │
              │  ANALYST      │ │  ANALYST   │ │  ANALYST     │
              │               │ │            │ │              │
              │ • Fibonacci   │ │ • KMI-30   │ │ • Backtest   │
              │ • RSI         │ │ • KMI All  │ │ • Sharpe     │
              │ • Elliott Wave│ │ • Compliance│ │ • Drawdown   │
              │ • MA Signals  │ │ • Watchlist │ │ • Win Rate   │
              │ • Charts      │ │ • Notes    │ │ • Position   │
              └───────────────┘ └────────────┘ └──────────────┘
                           │          │          │
                    ┌──────┴──────────┴──────────┴──────┐
                    │      SYNTHESIZED RESPONSE          │
                    │  Combined multi-agent analysis     │
                    └────────────────────────────────────┘
```

**Learn more:** [Architecture Documentation](docs/ARCHITECTURE.md)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- A free API key from [Groq](https://console.groq.com) (recommended) or another provider

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/agenticai-project.git
cd agenticai-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key
cp .env.example .env
# Edit .env — add your GROQ_API_KEY and set MODEL_PROVIDER=groq

# 4. Launch the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### Or use uv (recommended)

```bash
uv sync
streamlit run app.py
```

---

## 🎮 Usage Guide

### 1. Connect the Agent
1. Select your **LLM Provider** (Groq recommended — free, 14,400 requests/day)
2. Paste your **API key**
3. Choose **Agent Mode**:
   - 🌐 **Multi-Agent**: Supervisor + 3 specialists (recommended for thorough analysis)
   - 🔹 **Single Agent**: Classic ReAct agent (simpler, faster)
4. Click **Connect Agent**

### 2. Ask Questions
Try these example queries:
- *"Analyse WTL — should I buy?"*
- *"Is KEL Shariah-compliant?"*
- *"Backtest the strategy on UNITY"*
- *"List all Shariah-compliant low-price stocks"*
- *"What are the Fibonacci levels for CNERGY?"*

### 3. Explore Charts
Use the sidebar to select tickers and load interactive charts with Fibonacci overlays, RSI, volume, and swing points.

### 4. Run Backtests
Configure capital and lookback period, then run walk-forward backtests with detailed trade logs.

---

## 📖 Documentation

| Document | Description |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | System design, multi-agent workflow, data flow |
| [Strategy Guide](docs/STRATEGY.md) | Trading strategy logic, entry/exit rules, position sizing |
| [API Reference](docs/API.md) | Tool API, agent functions, configuration reference |
| [Contributing](CONTRIBUTING.md) | Development setup, coding standards, PR process |

---

## 🛠️ Project Structure

```
psx_multi_agent/
├── app.py                          # Streamlit UI (entry point)
├── config.py                       # Central configuration
├── Makefile                        # Common dev commands
├── pyproject.toml                  # Project metadata
├── requirements.txt                # Dependencies
├── .env.example                    # Environment template
│
├── agent/                          # AI Agent Package
│   ├── __init__.py                 # Exports (single + multi-agent)
│   ├── prompts.py                  # System prompts for all agents
│   ├── trading_agent.py            # Single ReAct agent
│   ├── specialized_agents.py       # 3 specialist sub-agents
│   └── supervisor_agent.py         # Multi-agent supervisor
│
├── tools/                          # Trading Tool Package
│   ├── __init__.py                 # All tools registry
│   ├── data_tools.py               # PSX data fetching
│   ├── technical_tools.py          # Fib/RSI/Elliott Wave/signals
│   ├── backtest_tools.py           # Walk-forward backtesting
│   ├── chart_tools.py              # Plotly visualizations
│   └── shariah_tools.py            # KMI compliance screening
│
├── docs/                           # Documentation
│   ├── ARCHITECTURE.md
│   ├── STRATEGY.md
│   └── API.md
│
├── .github/workflows/              # CI/CD
│   └── ci.yml
│
└── .gitignore
```

---

## 📊 Strategy Logic (Summary)

**Entry (BUY):**
- Price retraces to **50–61.8% Fibonacci** zone
- **RSI < 47** (below neutral — relaxed for PSX volatility)
- **Volume ≥ 0.8×** 20-day average

**Exit (TAKE PROFIT):**
- Price recovers to **23.6% Fibonacci** level

**Exit (STOP LOSS):**
- Price falls below **78.6% Fibonacci** level

**Position Sizing:**
- Risk = **2%** of current capital per trade
- Shares = `risk_amount / (entry - stop_loss)`
- Max 50% of capital in a single trade

**Full details:** [Strategy Guide](docs/STRATEGY.md)

---

## 🤖 Agent Modes

### Multi-Agent Mode (Default)
The **supervisor** classifies your query and delegates to the right specialist(s):
- **Technical Analyst** — Fibonacci, RSI, Elliott Wave, price signals
- **Shariah Analyst** — KMI-30 / KMI All-Share compliance screening
- **Risk Analyst** — Backtesting, risk metrics, position sizing

### Single Agent Mode
A single LangGraph ReAct agent with all tools. Simpler, faster — ideal for straightforward queries.

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## ⚠️ Disclaimer

**This tool is for educational and decision-support purposes only.**
It does **not** constitute financial advice. Always do your own research before investing.
Past backtest results do not guarantee future returns.
Verify Shariah compliance with certified Islamic finance scholars or the official PSX KMI index.

---

<p align="center">
  Built with ❤️ for the Agentic AI Course · Pakistan Stock Exchange
</p>
