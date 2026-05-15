"""API adapters — unified interface over Completions and Responses APIs."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from .config import ModelCapabilities
from .metrics import SingleRunMetrics

# ---------------------------------------------------------------------------
# Shared client factory
# ---------------------------------------------------------------------------

_client: OpenAI | None = None
_responses_client: OpenAI | None = None
_env_loaded = False

DEFAULT_AZURE_OPENAI_SCOPE = "https://cognitiveservices.azure.com/.default"
DEFAULT_FOUNDRY_SCOPE = "https://ai.azure.com/.default"


@dataclass(frozen=True)
class ClientSettings:
    base_url: str
    token_scope: str
    default_headers: dict[str, str] = field(default_factory=dict)
    default_query: dict[str, str] = field(default_factory=dict)


def _load_environment() -> None:
    global _env_loaded
    if not _env_loaded:
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", "API", ".env"))
        load_dotenv()  # also check cwd
        _env_loaded = True


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _normalise_openai_v1_base_url(endpoint: str) -> str:
    """Return a base URL that the OpenAI v1 client can append resources to."""
    endpoint = endpoint.strip().rstrip("/")
    parsed = urlsplit(endpoint)
    path = parsed.path.rstrip("/")
    lower_path = path.lower()
    if lower_path.endswith("/openai/v1"):
        normalised_path = path
    elif lower_path.endswith("/openai"):
        normalised_path = f"{path}/v1"
    else:
        normalised_path = f"{path}/openai/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, normalised_path, parsed.query, parsed.fragment))


def _looks_like_project_or_gateway_endpoint(endpoint: str) -> bool:
    parsed = urlsplit(endpoint)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    return (
        "/api/projects/" in path
        or host.endswith(".services.ai.azure.com")
        or host.endswith(".azure-api.net")
    )


def _gateway_subscription_key(endpoint: str) -> tuple[str | None, str]:
    key = _env_first(
        "AZURE_FOUNDRY_APIM_SUBSCRIPTION_KEY",
        "AZURE_FOUNDRY_GATEWAY_SUBSCRIPTION_KEY",
        "AZURE_FOUNDRY_GATEWAY_KEY",
        "APIM_SUBSCRIPTION_KEY",
    )
    if not key:
        return None, "header"

    location = (_env_first("AZURE_FOUNDRY_GATEWAY_KEY_LOCATION") or "").lower()
    if location in {"header", "query"}:
        return key, location

    # Foundry-managed APIM gateway URLs commonly validate the subscription key
    # at the route-matching layer, where the query-string form is accepted.
    return key, "query" if urlsplit(endpoint).netloc.lower().endswith(".azure-api.net") else "header"


def _client_settings(*, responses: bool = False) -> ClientSettings:
    _load_environment()

    if responses:
        endpoint = _env_first(
            "AZURE_FOUNDRY_RESPONSES_ENDPOINT",
            "AZURE_FOUNDRY_GATEWAY_ENDPOINT",
            "AZURE_FOUNDRY_PROJECT_ENDPOINT",
            "APIM_FOUNDRY_ENDPOINT",
            "AZURE_FOUNDRY_ENDPOINT",
            "OPENAI_BASE_URL",
        )
    else:
        endpoint = _env_first("AZURE_FOUNDRY_ENDPOINT", "OPENAI_BASE_URL")

    if not endpoint:
        raise EnvironmentError(
            "AZURE_FOUNDRY_ENDPOINT not set. Add it to your .env file."
        )

    base_url = _normalise_openai_v1_base_url(endpoint)
    token_scope = _env_first(
        "AZURE_FOUNDRY_RESPONSES_TOKEN_SCOPE" if responses else "AZURE_FOUNDRY_TOKEN_SCOPE",
        "AZURE_FOUNDRY_TOKEN_SCOPE",
    )
    if not token_scope:
        token_scope = (
            DEFAULT_FOUNDRY_SCOPE
            if responses and _looks_like_project_or_gateway_endpoint(base_url)
            else DEFAULT_AZURE_OPENAI_SCOPE
        )

    default_headers: dict[str, str] = {}
    default_query: dict[str, str] = {}
    if responses:
        subscription_key, location = _gateway_subscription_key(base_url)
        if subscription_key and location == "query":
            default_query["subscription-key"] = subscription_key
        elif subscription_key:
            default_headers["Ocp-Apim-Subscription-Key"] = subscription_key

    return ClientSettings(
        base_url=base_url,
        token_scope=token_scope,
        default_headers=default_headers,
        default_query=default_query,
    )


def _build_client(settings: ClientSettings) -> OpenAI:
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), settings.token_scope
    )
    return OpenAI(
        base_url=settings.base_url,
        api_key=token_provider,
        default_headers=settings.default_headers or None,
        default_query=settings.default_query or None,
    )


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = _build_client(_client_settings(responses=False))
    return _client


def _get_responses_client() -> OpenAI:
    global _responses_client
    if _responses_client is None:
        _responses_client = _build_client(_client_settings(responses=True))
    return _responses_client


def reset_client() -> None:
    """Force re-creation of the client (useful after env changes)."""
    global _client, _responses_client
    _client = None
    _responses_client = None


# ---------------------------------------------------------------------------
# Completions API adapter
# ---------------------------------------------------------------------------


def run_completions(
    model: str,
    system: str,
    user: str,
    *,
    stream: bool = True,
    max_tokens: int = 512,
    temperature: float = 0.7,
    timeout: float = 120.0,
    seed: int | None = None,
) -> SingleRunMetrics:
    """Execute a single Completions API call and return metrics."""
    client = _get_client()
    caps = ModelCapabilities.for_model(model)
    metrics = SingleRunMetrics(
        api_type="completions",
        prompt_label="",
        streaming=stream,
    )

    # o-series and gpt-5 use "developer" role instead of "system"
    messages = [
        {"role": caps.system_role, "content": system},
        {"role": "user", "content": user},
    ]

    # o-series doesn't support streaming on some models; honour caps
    if not caps.supports_streaming and stream:
        stream = False
        metrics.streaming = False

    wall_start = time.perf_counter()

    try:
        if stream:
            metrics = _completions_streaming(
                client, model, messages, max_tokens, temperature, timeout, metrics, wall_start, caps, seed=seed
            )
        else:
            metrics = _completions_non_streaming(
                client, model, messages, max_tokens, temperature, timeout, metrics, wall_start, caps, seed=seed
            )
    except Exception as exc:
        metrics.total_time = time.perf_counter() - wall_start
        metrics.end_to_end_latency = metrics.total_time
        metrics.success = False
        metrics.error = f"{type(exc).__name__}: {exc}"

    return metrics


def _completions_streaming(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout: float,
    metrics: SingleRunMetrics,
    wall_start: float,
    caps: ModelCapabilities,
    *,
    seed: int | None = None,
) -> SingleRunMetrics:
    first_token_received = False
    output_text_chunks: list[str] = []

    kwargs: dict = dict(
        model=model,
        messages=messages,
        max_completion_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        timeout=timeout,
    )
    if caps.supports_temperature:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed

    response_stream = client.chat.completions.create(**kwargs)

    for chunk in response_stream:
        if not first_token_received and chunk.choices and chunk.choices[0].delta.content:
            metrics.time_to_first_token = time.perf_counter() - wall_start
            first_token_received = True

        if chunk.choices and chunk.choices[0].delta.content:
            output_text_chunks.append(chunk.choices[0].delta.content)

        if chunk.choices and chunk.choices[0].finish_reason:
            metrics.finish_reason = chunk.choices[0].finish_reason

        # Usage comes in the final chunk when stream_options.include_usage=True
        if chunk.usage:
            metrics.input_tokens = chunk.usage.prompt_tokens
            metrics.output_tokens = chunk.usage.completion_tokens
            metrics.total_tokens = chunk.usage.total_tokens
            if hasattr(chunk.usage, "prompt_tokens_details") and chunk.usage.prompt_tokens_details:
                cached = getattr(chunk.usage.prompt_tokens_details, "cached_tokens", 0)
                metrics.cached_tokens = cached or 0
                metrics.is_cache_hit = (metrics.cached_tokens > 0)

        if hasattr(chunk, "model") and chunk.model:
            metrics.model_id = chunk.model
        if hasattr(chunk, "system_fingerprint") and chunk.system_fingerprint:
            metrics.system_fingerprint = chunk.system_fingerprint

    metrics.total_time = time.perf_counter() - wall_start
    metrics.end_to_end_latency = metrics.total_time
    metrics.response_text = "".join(output_text_chunks)
    if metrics.output_tokens and metrics.total_time > 0:
        metrics.tokens_per_second = metrics.output_tokens / metrics.total_time
    metrics.success = True
    return metrics


def _completions_non_streaming(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout: float,
    metrics: SingleRunMetrics,
    wall_start: float,
    caps: ModelCapabilities,
    *,
    seed: int | None = None,
) -> SingleRunMetrics:
    kwargs: dict = dict(
        model=model,
        messages=messages,
        max_completion_tokens=max_tokens,
        stream=False,
        timeout=timeout,
    )
    if caps.supports_temperature:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed

    response = client.chat.completions.create(**kwargs)

    metrics.total_time = time.perf_counter() - wall_start
    metrics.end_to_end_latency = metrics.total_time

    if response.usage:
        metrics.input_tokens = response.usage.prompt_tokens
        metrics.output_tokens = response.usage.completion_tokens
        metrics.total_tokens = response.usage.total_tokens
        if hasattr(response.usage, "prompt_tokens_details") and response.usage.prompt_tokens_details:
            cached = getattr(response.usage.prompt_tokens_details, "cached_tokens", 0)
            metrics.cached_tokens = cached or 0
            metrics.is_cache_hit = (metrics.cached_tokens > 0)

    if response.choices:
        metrics.finish_reason = response.choices[0].finish_reason or ""
        metrics.response_text = response.choices[0].message.content or ""

    if metrics.output_tokens and metrics.total_time > 0:
        metrics.tokens_per_second = metrics.output_tokens / metrics.total_time

    metrics.model_id = response.model or ""
    metrics.system_fingerprint = response.system_fingerprint or ""
    metrics.success = True
    return metrics


# ---------------------------------------------------------------------------
# Responses API adapter
# ---------------------------------------------------------------------------


def run_responses(
    model: str,
    system: str,
    user: str,
    *,
    stream: bool = True,
    max_tokens: int = 512,
    temperature: float = 0.7,
    timeout: float = 120.0,
    seed: int | None = None,
) -> SingleRunMetrics:
    """Execute a single Responses API call and return metrics."""
    caps = ModelCapabilities.for_model(model)
    metrics = SingleRunMetrics(
        api_type="responses",
        prompt_label="",
        streaming=stream,
    )

    client = _get_responses_client()
    # o-series doesn't support streaming on some models
    if not caps.supports_streaming and stream:
        stream = False
        metrics.streaming = False

    wall_start = time.perf_counter()

    try:
        if stream:
            metrics = _responses_streaming(
                client, model, system, user, max_tokens, temperature, timeout, metrics, wall_start, caps, seed=seed
            )
        else:
            metrics = _responses_non_streaming(
                client, model, system, user, max_tokens, temperature, timeout, metrics, wall_start, caps, seed=seed
            )
    except Exception as exc:
        metrics.total_time = time.perf_counter() - wall_start
        metrics.end_to_end_latency = metrics.total_time
        metrics.success = False
        metrics.error = f"{type(exc).__name__}: {exc}"

    return metrics


def _responses_streaming(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    metrics: SingleRunMetrics,
    wall_start: float,
    caps: ModelCapabilities,
    *,
    seed: int | None = None,
) -> SingleRunMetrics:
    first_token_received = False
    output_text_chunks: list[str] = []

    kwargs: dict = dict(
        model=model,
        instructions=system,
        input=user,
        max_output_tokens=max_tokens,
        stream=True,
        timeout=timeout,
    )
    if caps.supports_temperature:
        kwargs["temperature"] = temperature
    # Note: Responses API does not support the 'seed' parameter

    response_stream = client.responses.create(**kwargs)

    for event in response_stream:
        event_type = event.type if hasattr(event, "type") else ""

        # Detect first content delta as TTFT
        if not first_token_received and event_type == "response.output_text.delta":
            metrics.time_to_first_token = time.perf_counter() - wall_start
            first_token_received = True

        if event_type == "response.output_text.delta" and hasattr(event, "delta"):
            output_text_chunks.append(event.delta)

        # Final completed event carries usage
        if event_type == "response.completed" and hasattr(event, "response"):
            resp = event.response
            if hasattr(resp, "usage") and resp.usage:
                metrics.input_tokens = resp.usage.input_tokens
                metrics.output_tokens = resp.usage.output_tokens
                metrics.total_tokens = resp.usage.total_tokens
                if hasattr(resp.usage, "input_tokens_details") and resp.usage.input_tokens_details:
                    cached = getattr(resp.usage.input_tokens_details, "cached_tokens", 0)
                    metrics.cached_tokens = cached or 0
                    metrics.is_cache_hit = (metrics.cached_tokens > 0)
            metrics.model_id = getattr(resp, "model", "") or ""

    metrics.total_time = time.perf_counter() - wall_start
    metrics.end_to_end_latency = metrics.total_time
    metrics.response_text = "".join(output_text_chunks)
    if metrics.output_tokens and metrics.total_time > 0:
        metrics.tokens_per_second = metrics.output_tokens / metrics.total_time
    metrics.success = True
    return metrics


def _responses_non_streaming(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    metrics: SingleRunMetrics,
    wall_start: float,
    caps: ModelCapabilities,
    *,
    seed: int | None = None,
) -> SingleRunMetrics:
    kwargs: dict = dict(
        model=model,
        instructions=system,
        input=user,
        max_output_tokens=max_tokens,
        stream=False,
        timeout=timeout,
    )
    if caps.supports_temperature:
        kwargs["temperature"] = temperature
    # Note: Responses API does not support the 'seed' parameter

    response = client.responses.create(**kwargs)

    metrics.total_time = time.perf_counter() - wall_start
    metrics.end_to_end_latency = metrics.total_time

    if hasattr(response, "usage") and response.usage:
        metrics.input_tokens = response.usage.input_tokens
        metrics.output_tokens = response.usage.output_tokens
        metrics.total_tokens = response.usage.total_tokens
        if hasattr(response.usage, "input_tokens_details") and response.usage.input_tokens_details:
            cached = getattr(response.usage.input_tokens_details, "cached_tokens", 0)
            metrics.cached_tokens = cached or 0
            metrics.is_cache_hit = (metrics.cached_tokens > 0)

    if metrics.output_tokens and metrics.total_time > 0:
        metrics.tokens_per_second = metrics.output_tokens / metrics.total_time

    # Extract response text from the responses API output
    if hasattr(response, "output") and response.output:
        for item in response.output:
            if hasattr(item, "content") and item.content:
                for part in item.content:
                    if hasattr(part, "text"):
                        metrics.response_text += part.text

    metrics.model_id = getattr(response, "model", "") or ""
    metrics.success = True
    return metrics
