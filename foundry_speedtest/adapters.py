"""API adapters — unified interface over Completions and Responses APIs."""

from __future__ import annotations

import os
import time
from typing import Generator

from dotenv import load_dotenv
from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from .config import ModelCapabilities
from .metrics import SingleRunMetrics

# ---------------------------------------------------------------------------
# Shared client factory
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", "API", ".env"))
        load_dotenv()  # also check cwd
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        endpoint = os.getenv("AZURE_FOUNDRY_ENDPOINT")
        if not endpoint:
            raise EnvironmentError(
                "AZURE_FOUNDRY_ENDPOINT not set. Add it to your .env file."
            )
        _client = OpenAI(base_url=endpoint, api_key=token_provider)
    return _client


def reset_client() -> None:
    """Force re-creation of the client (useful after env changes)."""
    global _client
    _client = None


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
    client = _get_client()
    caps = ModelCapabilities.for_model(model)
    metrics = SingleRunMetrics(
        api_type="responses",
        prompt_label="",
        streaming=stream,
    )

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
