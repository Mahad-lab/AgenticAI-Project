"""
agent/__init__.py — PSX Trading Agent System

Provides two agent modes:
  - Single Agent: LangGraph ReAct agent with all tools (simple, fast)
  - Multi-Agent: Supervisor + specialized sub-agents (more thorough, role-based)
"""
from agent.trading_agent import get_llm, build_agent, run_query, stream_query
from agent.supervisor_agent import PSXTradingSupervisor
from agent.specialized_agents import (
    build_technical_analyst,
    build_shariah_analyst,
    build_risk_analyst,
    SPECIALIST_BUILDERS,
)

__all__ = [
    "get_llm",
    "build_agent",
    "run_query",
    "stream_query",
    "PSXTradingSupervisor",
    "build_technical_analyst",
    "build_shariah_analyst",
    "build_risk_analyst",
    "SPECIALIST_BUILDERS",
]
