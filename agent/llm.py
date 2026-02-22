"""
LLM Factory — returns the appropriate LLM client based on the LLM_PROVIDER
environment variable.

Supported providers:
  - claude        → langchain-anthropic (production)
  - groq          → langchain-openai with Groq endpoint (free tier dev)
  - ollama        → langchain-ollama (local)
  - openai_compat → langchain-openai with custom base URL

Default is 'claude' for production safety.
"""

import os

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def get_llm():
    """
    Returns the appropriate LangChain chat model based on LLM_PROVIDER env var.
    All providers use temperature=0 for deterministic, structured JSON output.
    """
    provider = os.getenv("LLM_PROVIDER", "claude")

    if provider == "claude":
        return ChatAnthropic(
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=1024,
            temperature=0,
        )

    elif provider == "ollama":
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            temperature=0,
        )

    elif provider == "groq":
        return ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
            temperature=0,
        )

    elif provider == "openai_compat":
        return ChatOpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL"),
            temperature=0,
        )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
