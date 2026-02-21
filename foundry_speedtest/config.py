"""Configuration, prompt sets, and constants for benchmarking."""

from dataclasses import dataclass, field

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
