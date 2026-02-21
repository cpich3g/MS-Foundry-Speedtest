"""Configuration, prompt sets, and constants for benchmarking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Model family detection — controls which API params are safe to send
# ---------------------------------------------------------------------------

def _is_o_series(model: str) -> bool:
    """Return True for reasoning models: o1, o1-mini, o1-preview, o3, o3-mini, o3-pro, o4-mini, etc."""
    m = model.lower()
    return bool(re.match(r"^o[0-9]", m))


def _is_gpt5(model: str) -> bool:
    """Return True for GPT-5 family models."""
    return model.lower().startswith("gpt-5")


@dataclass
class ModelCapabilities:
    """What a model family supports — drives parameter selection."""
    supports_temperature: bool = True
    supports_streaming: bool = True
    system_role: str = "system"       # "system" or "developer"
    max_tokens_key: str = "max_completion_tokens"  # param name for token limit

    @staticmethod
    def for_model(model: str) -> "ModelCapabilities":
        if _is_o_series(model):
            return ModelCapabilities(
                supports_temperature=False,
                supports_streaming=True,   # o3/o4 series support streaming; o1 may not but API returns clear error
                system_role="developer",
                max_tokens_key="max_completion_tokens",
            )
        if _is_gpt5(model):
            return ModelCapabilities(
                supports_temperature=True,
                supports_streaming=True,
                system_role="developer",
                max_tokens_key="max_completion_tokens",
            )
        # GPT-4.1, GPT-4o, GPT-4, etc.
        return ModelCapabilities(
            supports_temperature=True,
            supports_streaming=True,
            system_role="system",
            max_tokens_key="max_completion_tokens",
        )

# ---------------------------------------------------------------------------
# Prompt catalogue — diverse lengths & domains to stress-test fairly
# ---------------------------------------------------------------------------
BENCHMARK_PROMPTS: dict[str, dict] = {
    "short": {
        "system": "You are a concise assistant.",
        "user": "What is 2+2?",
        "label": "Short (trivial)",
    },
    "medium": {
        "system": "You are a helpful assistant.",
        "user": (
            "Explain the difference between TCP and UDP in networking. "
            "Cover reliability, ordering, use-cases, and performance trade-offs."
        ),
        "label": "Medium (technical)",
    },
    "long": {
        "system": "You are an expert technical writer.",
        "user": (
            "Write a detailed tutorial on building a REST API with Python and FastAPI. "
            "Cover project setup, routing, request validation with Pydantic, dependency injection, "
            "authentication with OAuth2, database integration with SQLAlchemy, error handling, "
            "testing with pytest, and deployment considerations. Include code examples for each section."
        ),
        "label": "Long (generation-heavy)",
    },
    "code": {
        "system": "You are an expert Python developer.",
        "user": "Write a Python function that implements a thread-safe LRU cache with TTL expiry.",
        "label": "Code generation",
    },
    "reasoning": {
        "system": "You are a logical reasoning expert.",
        "user": (
            "A farmer has a fox, a chicken, and a bag of grain. He needs to cross a river "
            "in a boat that can only carry him and one item at a time. If left alone, the fox "
            "will eat the chicken and the chicken will eat the grain. How does the farmer get "
            "everything across safely? Explain step by step."
        ),
        "label": "Reasoning / multi-step",
    },
}

# Prompt used for cache warm/cold testing (must be identical across runs)
CACHE_TEST_PROMPT = {
    "system": "You are a helpful assistant.",
    "user": "List the planets in our solar system in order from the sun.",
}


@dataclass
class BenchmarkConfig:
    """Runtime configuration for a benchmark session."""

    model: str = "gpt-4.1-nano"
    iterations: int = 3
    warmup: int = 1
    max_tokens: int = 512
    temperature: float = 0.7
    concurrency: int = 5
    prompt_keys: list[str] = field(default_factory=lambda: list(BENCHMARK_PROMPTS.keys()))
    stream: bool = True
    cache_rounds: int = 5
    timeout: float = 120.0
