"""
agent/supervisor_agent.py — Multi-Agent Supervisor for PSX Trading System

Architecture:
  Supervisor Agent (classifies + routes) → Specialized Sub-Agents → Synthesized Response

  - technical_analyst:  Fibonacci, RSI, Elliott Wave, price signals
  - shariah_analyst:    KMI Shariah-compliance screening
  - risk_analyst:       Backtesting, risk metrics, position sizing

The supervisor uses LangGraph's StateGraph to orchestrate a workflow:
  1. Receive user query
  2. Classify intent → route to appropriate specialist(s)
  3. Specialist agent executes with its focused tools
  4. (Optional) Multiple specialists for complex queries
  5. Return synthesised result
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Iterator, Sequence, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from agent.prompts import SUPERVISOR_PROMPT
from agent.specialized_agents import (
    SPECIALIST_BUILDERS, run_specialist,
)


# ─── Agent State ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    sub_agent_outputs: dict[str, str]
    user_query: str


# ─── Supervisor Node ────────────────────────────────────────────────────────────

ROLE_DESCRIPTIONS = {
    "technical": "Technical Analysis — Fibonacci, RSI, Elliott Wave, price signals, charts",
    "shariah":   "Shariah Compliance — KMI-30 / KMI All-Share screening, compliance check",
    "risk":      "Risk & Strategy — Backtesting, position sizing, risk metrics, trade logs",
}

ROLE_KEYWORDS = {
    "technical": ["price", "chart", "signal", "buy", "sell", "fibonacci",
                   "fib", "rsi", "trend", "technical", "analysis", "wave",
                   "support", "resistance", "level", "indicator"],
    "shariah":   ["shariah", "sharia", "halal", "haram", "compliant",
                   "compliance", "kmi", "islamic", "permissible", "screening"],
    "risk":      ["backtest", "risk", "strategy performance", "sharpe",
                   "drawdown", "win rate", "position size", "simulation",
                   "trade log", "profit factor"],
}


def _classify_intent(query: str) -> list[str]:
    """Classify user query into one or more specialist roles needed."""
    query_lower = query.lower()
    roles_needed = []
    for role, keywords in ROLE_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            roles_needed.append(role)
    return roles_needed if roles_needed else ["technical"]


def supervisor_node(state: AgentState, llm: BaseChatModel) -> dict:
    """Supervisor LLM node: classifies intent and updates state."""
    query = state["user_query"]
    roles = _classify_intent(query)

    # Build routing info
    selected = []
    for role in roles:
        desc = ROLE_DESCRIPTIONS.get(role, role)
        selected.append(f"  • {role}: {desc}")

    routing_msg = (
        f"📋 **Routing Analysis**\n\n"
        f"Query classified for: **{', '.join(roles)}**\n\n"
        f"Consulting specialist{'s' if len(roles) > 1 else ''}:\n"
        + "\n".join(selected)
    )

    return {
        "next": roles[0] if roles else "technical",
        "sub_agent_outputs": {},
        "messages": [HumanMessage(content=routing_msg)],
        "user_query": query,
    }


# ─── Specialist Sub-Agent Nodes ─────────────────────────────────────────────────

def _make_specialist_node(role: str):
    """Factory: create a graph node function for a specialist sub-agent."""

    def node_fn(state: AgentState, llm: BaseChatModel) -> dict:
        query = state["user_query"]
        chat_history = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant",
             "content": m.content}
            for m in state["messages"]
            if isinstance(m, (HumanMessage, AIMessage)) and m.content
        ]

        # Build/runspecialist agent
        builder = SPECIALIST_BUILDERS.get(role)
        if not builder:
            return {"sub_agent_outputs": {role: f"No builder found for {role}"}}

        agent = builder(llm)

        sub_prompt = f"[Specialist: {role}]\n{query}"
        result = run_specialist(agent, sub_prompt, chat_history[-6:] if len(chat_history) > 6 else chat_history)

        return {
            "sub_agent_outputs": {role: result},
            "messages": [AIMessage(content=f"[{role} analyst response received]")],
        }

    return node_fn


# ─── Router Logic ───────────────────────────────────────────────────────────────

def router(state: AgentState) -> str:
    """Route to the next specialist, or to 'synthesize' when all are done."""
    roles = _classify_intent(state["user_query"])
    completed = list(state["sub_agent_outputs"].keys())

    remaining = [r for r in roles if r not in completed]
    if remaining:
        return remaining[0]
    return "synthesize"


def synthesize_node(state: AgentState, llm: BaseChatModel) -> dict:
    """Synthesize all specialist outputs into a final response."""
    outputs = state.get("sub_agent_outputs", {})
    roles = _classify_intent(state["user_query"])

    synthesis_parts = [
        "## 📊 PSX Multi-Agent Analysis Report\n",
    ]

    for role in roles:
        output = outputs.get(role, "*No output received.*")
        synthesis_parts.append(f"### {ROLE_DESCRIPTIONS.get(role, role)}\n")
        synthesis_parts.append(output)
        synthesis_parts.append("")

    synthesis_parts.append(
        "---\n"
        "*⚠️ This analysis was generated by a multi-agent AI system. "
        "It is for decision-support purposes only and does not constitute financial advice. "
        "Always do your own research before investing.*"
    )

    final = "\n".join(synthesis_parts)
    return {
        "next": "end",
        "messages": [AIMessage(content=final)],
    }


# ─── Graph Builder ──────────────────────────────────────────────────────────────

class PSXTradingSupervisor:
    """
    Multi-agent supervisor that orchestrates specialized PSX trading agents.

    Usage:
        supervisor = PSXTradingSupervisor(llm)
        result = supervisor.run("Analyse WTL — should I buy?")
        for chunk in supervisor.stream("Analyse KEL"):
            print(chunk)
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.graph = self._build_graph()
        self._specialist_agents: dict[str, Any] = {}

    def _ensure_agent(self, role: str):
        """Lazily build specialist agents."""
        if role not in self._specialist_agents:
            builder = SPECIALIST_BUILDERS.get(role)
            if builder:
                self._specialist_agents[role] = builder(self.llm)

    def _wrap_node(self, fn):
        """Wrap a node function to inject llm."""

        def wrapped(state):
            return fn(state, llm=self.llm)
        return wrapped

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("supervisor", self._wrap_node(supervisor_node))
        for role in SPECIALIST_BUILDERS:
            workflow.add_node(role, self._wrap_node(_make_specialist_node(role)))
        workflow.add_node("synthesize", self._wrap_node(synthesize_node))

        # Edges
        workflow.add_conditional_edges(
            "supervisor",
            lambda s: s["next"],
            {r: r for r in SPECIALIST_BUILDERS},
        )
        for role in SPECIALIST_BUILDERS:
            workflow.add_conditional_edges(
                role,
                router,
                {r: r for r in SPECIALIST_BUILDERS} | {"synthesize": "synthesize"},
            )
        workflow.add_edge("synthesize", END)

        workflow.set_entry_point("supervisor")
        return workflow.compile()

    def run(self, query: str) -> str:
        """Run a query through the multi-agent system and return final result."""
        initial = {
            "messages": [HumanMessage(content=query)],
            "next": "technical",
            "sub_agent_outputs": {},
            "user_query": query,
        }
        result = self.graph.invoke(initial)
        final_msg = result["messages"][-1]
        if hasattr(final_msg, "content"):
            return str(final_msg.content) if not isinstance(final_msg.content, list) \
                else " ".join(b.get("text", "") for b in final_msg.content if isinstance(b, dict))
        return str(final_msg)

    def stream(self, query: str) -> Iterator[str]:
        """Stream query results token-wise."""
        initial = {
            "messages": [HumanMessage(content=query)],
            "next": "technical",
            "sub_agent_outputs": {},
            "user_query": query,
        }

        full_text = ""
        for event in self.graph.stream(initial, stream_mode="values"):
            last_msg = event.get("messages", [None])[-1] if event.get("messages") else None
            if last_msg and hasattr(last_msg, "content") and isinstance(last_msg, AIMessage):
                content = last_msg.content
                if isinstance(content, list):
                    text = " ".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                else:
                    text = str(content)
                if text and text != full_text:
                    yield text
                    full_text = text
