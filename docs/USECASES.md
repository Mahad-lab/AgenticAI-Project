# Use Cases: Single vs Multi-Agent Mode

This document provides practical use cases you can run in both **Single Agent** and **Multi-Agent** modes to understand the difference. Switch modes in the sidebar → Agent Mode.

---

## Use Case 1: Single Stock Analysis

**"Analyse WTL — should I buy?"**

### Single Agent
One ReAct agent handles everything. Fast response, direct answer.

```
You → "Analyse WTL — should I buy?"

Agent calls: get_stock_summary("WTL")
Agent calls: analyze_technical("WTL")
Agent calls: check_shariah("WTL")

Response: Basic signal + compliance status
```

### Multi-Agent
Supervisor classifies as "technical + shariah", delegates to 2 specialists sequentially.

```
You → "Analyse WTL — should I buy?"

Supervisor: "Query classified for: technical, shariah"
  → Technical Analyst: get_stock_summary + analyze_technical
  → Shariah Analyst: check_shariah
  → Synthesizer combines both

Response: Structured report with:
  - Technical Analysis section (Fib levels, RSI, signal, stop-loss, TP)
  - Shariah Compliance section (KMI status, price, notes)
```

---

## Use Case 2: Shariah Compliance Check

**"Is KEL Shariah-compliant?"**

### Single Agent
```
You → "Is KEL Shariah-compliant?"

Agent calls: check_shariah("KEL")
Response: Compliance status + note
```

### Multi-Agent
```
You → "Is KEL Shariah-compliant?"

Supervisor: "Query classified for: shariah"
  → Shariah Analyst: check_shariah("KEL")
  → Direct to synthesize (only 1 specialist needed)

Response: Same result, but clearly tagged as coming from the Shariah Analyst
```

---

## Use Case 3: Multi-Domain Query (1)

**"Analyse UNITY — check compliance and run a backtest"**

### Single Agent
Single agent context-switches between domains. May produce a less structured response.

```
You → "Analyse UNITY — check compliance and run a backtest"

Agent calls: get_stock_summary("UNITY")
Agent calls: analyze_technical("UNITY")
Agent calls: check_shariah("UNITY")
Agent calls: run_backtest("UNITY")

Response: Combined text output, agent decides the structure
```

### Multi-Agent
Supervisor correctly classifies as needing 3 specialists, runs them in sequence, produces a structured report.

```
You → "Analyse UNITY — check compliance and run a backtest"

Supervisor: "Query classified for: technical, shariah, risk"
  → Technical Analyst: get_stock_summary + analyze_technical
  → Shariah Analyst: check_shariah
  → Risk Analyst: run_backtest
  → Synthesizer: Combines all 3 sections

Response:
  📊 Technical Analysis — UNITY
    Signal: BUY | RSI: 35.2 | Fib levels | Elliott Wave

  🕌 Shariah Compliance — UNITY
    Status: COMPLIANT | KMI-30: Yes | Price: 8.50 PKR

  📈 Backtest Report — UNITY
    Return: +12.3% | Win Rate: 58% | Sharpe: 1.2 | 15 trades
```

---

## Use Case 4: Backtest Request

**"Backtest the strategy on KEL with 10,000 PKR"**

### Single Agent
```
You → "Backtest the strategy on KEL with 10,000 PKR"

Agent calls: run_backtest("KEL,365,10000")
Response: Backtest metrics + trade log summary
```

### Multi-Agent
```
You → "Backtest the strategy on KEL with 10,000 PKR"

Supervisor: "Query classified for: risk"
  → Risk Analyst: run_backtest("KEL,365,10000")
  → Direct to synthesize

Response: Risk-focused report (same data, tagged section)
```

---

## Use Case 5: Watchlist Exploration

**"List all Shariah-compliant low-price stocks"**

### Single Agent
```
You → "List all Shariah-compliant low-price stocks"

Agent calls: list_shariah_watchlist()
Response: Formatted list of compliant stocks
```

### Multi-Agent
```
You → "List all Shariah-compliant low-price stocks"

Supervisor: "Query classified for: shariah"
  → Shariah Analyst: list_shariah_watchlist()
  → Direct to synthesize

Response: Same list, tagged as Shariah Analyst output
```

---

## Use Case 6: Complex Multi-Query (Best Multi-Agent Demo)

**"Analyse CNERGY — Is it Shariah-compliant? Should I buy? And how did the strategy perform historically?"**

### Single Agent
The single agent may drop some context or produce a less structured answer because it handles all tools in one flat sequence.

```
You → "Analyse CNERGY — compliance, buy signal, backtest"

Agent calls: get_stock_summary, analyze_technical, check_shariah, run_backtest
Response: Mixed output, may miss completeness
```

### Multi-Agent
```
You → "Analyse CNERGY — compliance, buy signal, backtest"

Supervisor: "Query classified for: technical, shariah, risk"
  Routing Analysis:
    • technical: Technical Analysis — Fibonacci, RSI, Elliott Wave, price signals
    • shariah: Shariah Compliance — KMI-30 / KMI All-Share screening
    • risk: Risk & Strategy — Backtesting, position sizing, risk metrics

[Technical Analyst response received]
[Shariah Analyst response received]
[Risk Analyst response received]

## 📊 PSX Multi-Agent Analysis Report

### Technical Analysis — CNERGY
  Signal: WATCH
  Reasoning: Price (4.65) approaching buy zone (4.12–4.48)...
  Fibonacci levels: ...
  RSI: 38.2 (approaching oversold)
  Elliott Wave: BULLISH_IMPULSE

### Shariah Compliance — CNERGY
  Status: COMPLIANT
  In KMI-30: Yes
  Price: 4.65 PKR (✅ LOW-PRICE)
  Note: Cnergyico PK — KMI All-Share compliant.

### Risk & Strategy — CNERGY
  Start Capital: 5,000.00 PKR
  End Capital: 5,823.40 PKR
  Total Return: +16.47%
  Win Rate: 62.5%
  Sharpe Ratio: 1.35
  Total Trades: 8
  Recent trades: ...
```

---

## Summary Table

| Aspect | Single Agent | Multi-Agent |
|---|---|---|
| **Latency** | Fast (~2-5s) | Slower (~5-15s, multiple rounds) |
| **Structure** | Free-form text | Sectioned, labelled report |
| **Specialization** | One prompt for everything | Each agent has focused expertise |
| **Auditability** | Hard to trace which tool produced what | Clear which specialist generated each section |
| **Multi-domain queries** | May drop context | Handles systematically |
| **Simple queries** | ✅ Excellent | ✅ Good (slight overhead) |
| **Complex queries** | ⚠️ Average | ✅ Excellent |

## Recommendation

- Use **Single Agent** for: quick lookups, simple questions, testing
- Use **Multi-Agent** for: thorough analysis, multi-domain queries, presentations, demos
