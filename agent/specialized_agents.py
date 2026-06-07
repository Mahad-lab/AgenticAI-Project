"""
agent/specialized_agents.py — Specialized PSX sub-agents for the multi-agent system
Each sub-agent has a focused role and its own set of tools.
"""
from __future__ import annotations

import os
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool

from agent.prompts import TECHNICAL_ANALYST_PROMPT, SHARIAH_ANALYST_PROMPT, RISK_ANALYST_PROMPT
from tools.data_tools import get_stock_summary
from tools.technical_tools import analyze_technical
from tools.shariah_tools import check_shariah, list_shariah_watchlist
from tools.backtest_tools import run_backtest


def _get_tools_for_role(role: str) -> list[BaseTool]:
    tools_map = {
        "technical": [get_stock_summary, analyze_technical],
        "shariah":   [check_shariah, list_shariah_watchlist],
        "risk":      [run_backtest],
    }
    return tools_map.get(role, [])


def build_technical_analyst(llm: BaseChatModel):
    """Build the Technical Analyst sub-agent (Fibonacci, RSI, Elliott Wave, signals)."""
    return create_agent(
        model=llm,
        tools=_get_tools_for_role("technical"),
        system_prompt=SystemMessage(content=TECHNICAL_ANALYST_PROMPT),
    )


def build_shariah_analyst(llm: BaseChatModel):
    """Build the Shariah Compliance Analyst sub-agent (KMI screening)."""
    return create_agent(
        model=llm,
        tools=_get_tools_for_role("shariah"),
        system_prompt=SystemMessage(content=SHARIAH_ANALYST_PROMPT),
    )


def build_risk_analyst(llm: BaseChatModel):
    """Build the Risk & Strategy Analyst sub-agent (backtesting, risk metrics)."""
    return create_agent(
        model=llm,
        tools=_get_tools_for_role("risk"),
        system_prompt=SystemMessage(content=RISK_ANALYST_PROMPT),
    )


SPECIALIST_BUILDERS = {
    "technical": build_technical_analyst,
    "shariah":   build_shariah_analyst,
    "risk":      build_risk_analyst,
}


def run_specialist(agent, user_message: str,
                   chat_history: list[dict] | None = None) -> str:
    """Run a single query through a specialist agent and return text response."""
    messages = []
    for msg in (chat_history or []):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_message))

    result = agent.invoke({"messages": messages})
    final_msg = result["messages"][-1]
    if hasattr(final_msg, "content"):
        content = final_msg.content
        if isinstance(content, list):
            return " ".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return str(content)
    return str(final_msg)
