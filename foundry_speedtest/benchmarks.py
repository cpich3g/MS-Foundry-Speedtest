"""Benchmark orchestrator — runs test suites and collects metrics."""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable

from .adapters import run_completions, run_responses
from .config import BENCHMARK_PROMPTS, CACHE_TEST_PROMPT, VARIABILITY_PROMPT, BenchmarkConfig
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


# ---------------------------------------------------------------------------
# Variability / determinism test
# ---------------------------------------------------------------------------


@dataclass
class VariabilityResult:
    """Results from a single variability test phase (seeded or unseeded)."""
    api_type: str
    seeded: bool
    seed: int | None
    runs: list[SingleRunMetrics] = field(default_factory=list)
    pairwise_similarities: list[float] = field(default_factory=list)
    fingerprints: list[str] = field(default_factory=list)

    @property
    def avg_similarity(self) -> float:
        return sum(self.pairwise_similarities) / len(self.pairwise_similarities) if self.pairwise_similarities else 0.0

    @property
    def min_similarity(self) -> float:
        return min(self.pairwise_similarities) if self.pairwise_similarities else 0.0

    @property
    def max_similarity(self) -> float:
        return max(self.pairwise_similarities) if self.pairwise_similarities else 0.0

    @property
    def fingerprint_consistent(self) -> bool:
        unique = set(f for f in self.fingerprints if f)
        return len(unique) <= 1

    @property
    def verdict(self) -> str:
        avg = self.avg_similarity
        if avg >= 0.95:
            return "Deterministic"
        if avg >= 0.75:
            return "Mostly deterministic"
        if avg >= 0.50:
            return "Semi-variable"
        return "Non-deterministic"


def _text_similarity(a: str, b: str) -> float:
    """Compute similarity ratio between two texts using SequenceMatcher."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def benchmark_variability(
    api_type: str,
    cfg: BenchmarkConfig,
    *,
    on_run: Callable[[int, str, SingleRunMetrics], None] | None = None,
) -> tuple[VariabilityResult, VariabilityResult]:
    """Run variability test: N calls without seed, then N calls with seed.

    Returns (unseeded_result, seeded_result).
    """
    runner = API_RUNNERS[api_type]
    prompt = VARIABILITY_PROMPT

    results: list[VariabilityResult] = []

    for seeded in (False, True):
        seed_val = cfg.variability_seed if seeded else None
        vr = VariabilityResult(
            api_type=api_type,
            seeded=seeded,
            seed=seed_val,
        )

        for i in range(cfg.variability_rounds):
            m = runner(
                cfg.model,
                prompt["system"],
                prompt["user"],
                stream=False,  # non-streaming for consistent text capture
                max_tokens=cfg.variability_max_tokens,
                temperature=cfg.temperature,
                timeout=cfg.timeout,
                seed=seed_val,
            )
            m.prompt_label = f"Variability ({'seed=' + str(seed_val) if seeded else 'no seed'})"
            vr.runs.append(m)
            if m.system_fingerprint:
                vr.fingerprints.append(m.system_fingerprint)
            if on_run:
                on_run(i, "seeded" if seeded else "unseeded", m)

        # Compute pairwise similarities between all successful runs
        texts = [r.response_text for r in vr.runs if r.success and r.response_text]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                vr.pairwise_similarities.append(_text_similarity(texts[i], texts[j]))

        results.append(vr)

    return results[0], results[1]
