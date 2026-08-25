"""
One client, three lanes.

Same idea as the CSE476 course lanes: Groq, Microsoft Foundry, or local Ollama.
The rest of the project only calls get_client() / get_model().
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


class LaneError(RuntimeError):
    """Lane selected but not configured. The message says how to fix it."""


@dataclass(frozen=True)
class Lane:
    key: str
    name: str
    base_url: str | None
    key_env: str
    default_model: str
    note: str
    retired: bool = False


LANES: dict[str, Lane] = {
    "groq": Lane(
        key="groq",
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        key_env="GROQ_API_KEY",
        default_model="openai/gpt-oss-20b",
        note="Free default. Get a key at console.groq.com/keys.",
    ),
    "foundry": Lane(
        key="foundry",
        name="Microsoft Foundry",
        base_url=None,
        key_env="AZURE_OPENAI_API_KEY",
        default_model="chat-demo",
        note="Set AZURE_OPENAI_ENDPOINT (ending in /openai/v1/) and AZURE_OPENAI_API_KEY.",
    ),
    "local": Lane(
        key="local",
        name="Ollama (local)",
        base_url="http://localhost:11434/v1",
        key_env="",
        default_model="llama3.2",
        note="No key. Run `ollama serve` and `ollama pull llama3.2`.",
    ),
    "github": Lane(
        key="github",
        name="GitHub Models (retired)",
        base_url="https://models.github.ai/inference",
        key_env="GITHUB_TOKEN",
        default_model="openai/gpt-4.1-mini",
        note="Retired 30 July 2026. Use groq or local.",
        retired=True,
    ),
}

PROVIDER = os.getenv("PROVIDER", "groq").strip().lower()


def _lane(provider: str | None = None) -> Lane:
    key = (provider or PROVIDER).strip().lower()
    if key not in LANES:
        raise LaneError(
            f"PROVIDER={key!r} is not a lane. Use groq, foundry, or local."
        )
    lane = LANES[key]
    if lane.retired:
        raise LaneError(
            f"{lane.name} is no longer available. Set PROVIDER=groq or PROVIDER=local."
        )
    return lane


def get_model(provider: str | None = None) -> str:
    override = os.getenv("MODEL", "").strip()
    if override:
        return override
    return _lane(provider).default_model


def get_client(provider: str | None = None):
    """OpenAI-compatible client for the active lane."""
    from openai import OpenAI

    lane = _lane(provider)

    if lane.key == "foundry":
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        api_key = os.getenv(lane.key_env, "").strip()
        if not endpoint or not api_key:
            raise LaneError(
                "Foundry is selected but AZURE_OPENAI_ENDPOINT or "
                "AZURE_OPENAI_API_KEY is missing. Or set PROVIDER=groq."
            )
        base = endpoint.rstrip("/")
        for suffix in ("/responses", "/chat/completions", "/completions"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        if not base.endswith("/openai/v1"):
            base = base + "/openai/v1"
        return OpenAI(base_url=base + "/", api_key=api_key, timeout=60.0)

    if lane.key == "local":
        return OpenAI(base_url=lane.base_url, api_key="ollama", timeout=180.0)

    api_key = os.getenv(lane.key_env, "").strip()
    if not api_key:
        raise LaneError(
            f"{lane.name} is selected but {lane.key_env} is not set. "
            f"Add it to .env, or run the notebook in offline mode."
        )
    return OpenAI(base_url=lane.base_url, api_key=api_key, timeout=60.0)


def describe() -> str:
    lane = _lane()
    return f"Lane: {lane.name} ({lane.key})  |  Model: {get_model()}"


def lane_is_configured(provider: str | None = None) -> bool:
    try:
        get_client(provider)
        return True
    except LaneError:
        return False
