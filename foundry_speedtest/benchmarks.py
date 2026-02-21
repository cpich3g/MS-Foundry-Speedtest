"""Benchmark orchestrator — runs test suites and collects metrics."""

from __future__ import annotations

import concurrent.futures
import time
from typing import Callable

from .adapters import run_completions, run_responses
from .config import BENCHMARK_PROMPTS, CACHE_TEST_PROMPT, BenchmarkConfig
from .metrics import AggregateMetrics, SingleRunMetrics


RunnerFn = Callable[..., SingleRunMetrics]

API_RUNNERS: dict[str, RunnerFn] = {
    "completions": run_completions,
    "responses": run_responses,
}


def _single_call(
    runner: RunnerFn,
    model: str,
    system: str,
    user: str,
    stream: bool,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> SingleRunMetrics:
    return runner(
        model, system, user,
        stream=stream,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Per-prompt benchmark
# ---------------------------------------------------------------------------


def benchmark_prompt(
    api_type: str,
    prompt_key: str,
    cfg: BenchmarkConfig,
    *,
    stream: bool,
    on_run: Callable[[int, SingleRunMetrics], None] | None = None,
) -> AggregateMetrics:
    """Run N iterations of a single prompt and aggregate."""
    prompt = BENCHMARK_PROMPTS[prompt_key]
    runner = API_RUNNERS[api_type]
    agg = AggregateMetrics(
        api_type=api_type,
        prompt_label=prompt["label"],
        streaming=stream,
    )

    # Warmup
    for _ in range(cfg.warmup):
        _single_call(
            runner, cfg.model, prompt["system"], prompt["user"],
            stream, cfg.max_tokens, cfg.temperature, cfg.timeout,
        )

    # Measured runs
    for i in range(cfg.iterations):
        m = _single_call(
            runner, cfg.model, prompt["system"], prompt["user"],
            stream, cfg.max_tokens, cfg.temperature, cfg.timeout,
        )
        m.prompt_label = prompt["label"]
        agg.runs.append(m)
        if on_run:
            on_run(i, m)

    return agg


# ---------------------------------------------------------------------------
# Cache test — repeated identical prompts to measure cache warm-up
# ---------------------------------------------------------------------------


def benchmark_cache(
    api_type: str,
    cfg: BenchmarkConfig,
    *,
    on_run: Callable[[int, SingleRunMetrics], None] | None = None,
) -> AggregateMetrics:
    """Send the same prompt N times to observe caching behaviour."""
    runner = API_RUNNERS[api_type]
    agg = AggregateMetrics(
        api_type=api_type,
        prompt_label="Cache test",
        streaming=True,
    )

    for i in range(cfg.cache_rounds):
        m = _single_call(
            runner, cfg.model,
            CACHE_TEST_PROMPT["system"],
            CACHE_TEST_PROMPT["user"],
            True, cfg.max_tokens, cfg.temperature, cfg.timeout,
        )
        m.prompt_label = "Cache test"
        agg.runs.append(m)
        if on_run:
            on_run(i, m)

    return agg


# ---------------------------------------------------------------------------
# Concurrency / throughput test
# ---------------------------------------------------------------------------


def benchmark_concurrency(
    api_type: str,
    cfg: BenchmarkConfig,
    *,
    on_batch_done: Callable[[list[SingleRunMetrics]], None] | None = None,
) -> AggregateMetrics:
    """Fire N concurrent requests and measure overall throughput."""
    runner = API_RUNNERS[api_type]
    prompt = BENCHMARK_PROMPTS["medium"]
    agg = AggregateMetrics(
        api_type=api_type,
        prompt_label=f"Concurrency x{cfg.concurrency}",
        streaming=False,  # non-streaming is simpler to measure concurrently
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        futures = [
            pool.submit(
                _single_call,
                runner, cfg.model,
                prompt["system"], prompt["user"],
                False, cfg.max_tokens, cfg.temperature, cfg.timeout,
            )
            for _ in range(cfg.concurrency)
        ]
        results = []
        for fut in concurrent.futures.as_completed(futures):
            m = fut.result()
            m.prompt_label = f"Concurrency x{cfg.concurrency}"
            agg.runs.append(m)
            results.append(m)

    if on_batch_done:
        on_batch_done(results)

    return agg
