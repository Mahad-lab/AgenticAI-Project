# Architecture Guide

## System Overview

The PSX Multi-Agent Trading System is a collaborative AI platform that combines
multi-agent orchestration, technical analysis, and Islamic finance compliance
for the Pakistan Stock Exchange.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           User Interface (Streamlit)                             │
│  ┌───────────────────────┐  ┌────────────────────────────────────────────────┐   │
│  │    Sidebar Config     │  │              Main Area (2-col)                 │   │
│  │  • LLM Provider       │  │  ┌──────────────┐  ┌────────────────────────┐ │   │
│  │  • API Key            │  │  │  Chat Column  │  │     Chart Tabs        │ │   │
│  │  • Agent Mode         │  │  │  (AI Agent)   │  │  ┌─────┬──────┬────┐ │ │   │
│  │  • Ticker/Capital     │  │  │               │  │  │Chart│BTest │Cmp │ │ │   │
│  │  • Quick Select       │  │  └──────────────┘  │  └─────┴──────┴────┘ │ │   │
│  └───────────────────────┘  └────────────────────────────────────────────────┘   │
└──────────────────────────────────┬───────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         Multi-Agent Orchestration                                │
│                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────┐      │
│   │                     SUPERVISOR AGENT                                 │      │
│   │  ┌────────────────────────────────────────────────────────────────┐  │      │
│   │  │  Intent Classifier (keyword-based role matching)              │  │      │
│   │  │  Router (condition edge in StateGraph)                       │  │      │
│   │  │  Synthesizer (combines specialist outputs)                    │  │      │
│   │  └────────────────────────────────────────────────────────────────┘  │      │
│   └────────────────────────────────┬─────────────────────────────────────┘      │
│                                     │                                            │
│           ┌─────────────────────────┼────────────────────────────┐               │
│           │                         │                            │               │
│           ▼                         ▼                            ▼               │
│   ┌───────────────┐     ┌────────────────────┐     ┌────────────────────┐       │
│   │  TECHNICAL    │     │   SHARIAH           │     │   RISK             │       │
│   │  ANALYST      │     │   ANALYST           │     │   ANALYST          │       │
│   │               │     │                     │     │                    │       │
│   │  Tools:       │     │  Tools:             │     │  Tools:            │       │
│   │  • stock_sum  │     │  • check_shariah    │     │  • run_backtest    │       │
│   │  • technical  │     │  • list_watchlist   │     │  • (no others)     │       │
│   └───────────────┘     └────────────────────┘     └────────────────────┘       │
│                           │                      │                               │
│                           └──────────────────────┘                               │
│                                              │                                   │
│                                              ▼                                   │
│                                ┌──────────────────────┐                         │
│                                │   Synthesized Report  │                         │
│                                └──────────────────────┘                         │
└──────────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              Data Layer                                          │
│                                                                                  │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐  ┌────────────────┐     │
│  │  psxdata     │  │  technical_    │  │  backtest_   │  │  shariah_     │     │
│  │  (OHLCV API) │  │  tools.py     │  │  tools.py    │  │  tools.py     │     │
│  │              │  │  (Fib/RSI/    │  │  (engine)    │  │  (KMI data)   │     │
│  │              │  │   Wave)       │  │              │  │               │     │
│  └──────────────┘  └────────────────┘  └──────────────┘  └────────────────┘     │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    chart_tools.py (Plotly)                               │    │
│  │  Candlestick + Fib overlay + RSI + Volume + Swing points + Trade signals │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Multi-Agent Workflow

### 1. Intent Classification (`supervisor_agent.py:_classify_intent`)
When a user sends a query, the supervisor classifies it by scanning for keywords:

| Role | Keywords |
|---|---|
| **technical** | price, chart, signal, buy, sell, fibonacci, fib, rsi, trend, wave, support, resistance |
| **shariah** | shariah, halal, haram, compliant, kmi, islamic, permissible |
| **risk** | backtest, risk, sharpe, drawdown, win rate, position size, simulation |

A query can match **multiple roles** (e.g., "Analyse WTL for Shariah compliance and backtest it").

### 2. Routing
The supervisor routes to the first specialist. After that specialist completes,
the `router()` function checks if more specialists are needed and routes accordingly.

### 3. Specialist Execution
Each specialist runs as an independent LangGraph ReAct agent with:
- A focused system prompt (no distractions from unrelated tools)
- A subset of tools relevant to its domain
- Its own conversation context

### 4. Synthesis
Once all specialists have reported, the `synthesize_node` combines outputs into
a structured report with clearly labeled sections.

---

## Data Flow

```
User Query
    │
    ▼
┌──────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Supervisor   │────▶│  Specialist N    │────▶│  Synthesizer      │──▶ Response
│  (classify)   │     │  (tool calls)    │     │  (combine)        │
└──────────────┘     └──────────────────┘     └───────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  LangChain   │
                    │  @tool fn    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Python      │
                    │  business    │
                    │  logic       │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  External    │
                    │  APIs        │
                    │  (psxdata)   │
                    └──────────────┘
```

---

## Agent State

The multi-agent system uses LangGraph's `StateGraph` with this state schema:

```python
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]     # Full conversation history
    next: str                           # Next agent to route to
    sub_agent_outputs: dict[str, str]   # Collected specialist results
    user_query: str                     # Original query (for routing)
```

---

## Key Design Decisions

### Why Multi-Agent?
1. **Role specialization** — Each agent has focused expertise, reducing hallucination
2. **Auditability** — Clear which agent produced which analysis
3. **Scalability** — Easy to add new specialist agents (e.g., News Analyst, Fundamental Analyst)
4. **Fault isolation** — One agent failure doesn't crash the whole system

### Why LangGraph?
- Native support for cyclic graphs (agents calling tools repeatedly)
- Built-in state management
- Streaming support for real-time UI updates
- Easy to add conditional routing

### Why Single Agent Mode?
- Lower latency for simple queries
- No overhead from multi-step orchestration
- Familiar ReAct pattern for debugging

---

## Configuration

All tunable parameters live in `config.py`:

| Category | Parameters |
|---|---|
| **LLM** | Provider, model names, temperature, API key env vars |
| **Trading** | Default capital (5,000 PKR), risk per trade (2%), commission (0.1%) |
| **Technical** | Fib levels, RSI period/thresholds, MA periods, swing window |
| **Shariah** | Watchlist tickers, KMI-30 constituents, non-compliant list |
| **Data** | Default lookback (365 days), swing detection window (5 bars) |
