"""
agent/trading_agent.py — LangGraph ReAct agent with pluggable LLM providers
"""
from __future__ import annotations

import os
from typing import Any, Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from agent.prompts import SYSTEM_PROMPT
from tools import ALL_TOOLS


# ─── LLM Factory ─────────────────────────────────────────────────────────────

def get_llm(provider: str | None = None, api_key: str | None = None) -> BaseChatModel:
    """
    Return a LangChain chat model for the given provider.

    provider: "groq" | "gemini" | "openai_compatible"
    api_key:  override the env-variable key (useful for Streamlit secrets)
    """
    from config import MODEL_PROVIDER, PROVIDER_CONFIGS

    provider = (provider or MODEL_PROVIDER).lower()
    cfg      = PROVIDER_CONFIGS.get(provider)
    if not cfg:
        raise ValueError(f"Unknown provider '{provider}'. "
                         "Choose: groq | gemini | openai_compatible")

    resolved_key = api_key or os.getenv(cfg["api_key_env"], "")
    if not resolved_key:
        raise EnvironmentError(
            f"API key not found. Set {cfg['api_key_env']} in your .env file "
            f"or pass it directly to get_llm()."
        )

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=cfg["model"],
            temperature=cfg["temperature"],
            groq_api_key=resolved_key,
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=cfg["model"],
            temperature=cfg["temperature"],
            google_api_key=resolved_key,
            convert_system_message_to_human=True,  # Gemini quirk
        )

    elif provider == "openai_compatible":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=cfg["model"],
            temperature=cfg["temperature"],
            openai_api_key=resolved_key,
            base_url=cfg.get("base_url"),
        )

    raise ValueError(f"Unsupported provider: {provider}")


# ─── Agent Builder ────────────────────────────────────────────────────────────

def build_agent(llm: BaseChatModel):
    """
    Build a LangGraph ReAct agent with all trading tools.
    Returns a compiled LangGraph graph.
    """
    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        state_modifier=SYSTEM_PROMPT,
    )
    return agent


# ─── Synchronous Query Runner ─────────────────────────────────────────────────

def run_query(agent, user_message: str,
              chat_history: list[dict] | None = None) -> str:
    """
    Run a single query through the agent and return the final text response.

    chat_history: list of {"role": "user"|"assistant", "content": str}
    """
    messages = []
    for msg in (chat_history or []):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_message))

    result = agent.invoke({"messages": messages})

    # Extract the final AI message content
    final_msg = result["messages"][-1]
    if hasattr(final_msg, "content"):
        content = final_msg.content
        # Handle list content (some providers return [{type:text, text:...}])
        if isinstance(content, list):
            return " ".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return str(content)
    return str(final_msg)


# ─── Streaming Runner (for Streamlit) ────────────────────────────────────────

def stream_query(agent, user_message: str,
                 chat_history: list[dict] | None = None) -> Iterator[str]:
    """
    Stream the agent's response token by token.
    Yields text chunks as they arrive.

    Usage in Streamlit:
        for chunk in stream_query(agent, user_input, history):
            response_placeholder.markdown(chunk)
    """
    messages = []
    for msg in (chat_history or []):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_message))

    full_text = ""
    for event in agent.stream({"messages": messages}, stream_mode="values"):
        last_msg = event["messages"][-1]
        # Only yield AI text messages (skip tool calls)
        if hasattr(last_msg, "content") and isinstance(last_msg, AIMessage):
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
