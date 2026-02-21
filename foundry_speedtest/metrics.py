"""Metrics dataclass and statistics helpers."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass
class SingleRunMetrics:
    """Metrics captured from a single API call."""

    api_type: str  # "completions" or "responses"
    prompt_label: str
    streaming: bool

    # Timing (seconds)
    time_to_first_token: float | None = None  # None for non-streaming
    total_time: float = 0.0
    end_to_end_latency: float = 0.0  # includes overhead

    # Tokens
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Throughput
    tokens_per_second: float = 0.0

    # Cache
    cached_tokens: int = 0
    is_cache_hit: bool = False

    # Status
    success: bool = True
    error: str | None = None

    # Model info echoed back
    model_id: str = ""

    # Response metadata
    finish_reason: str = ""
    system_fingerprint: str = ""

    # Response text (populated for variability tests)
    response_text: str = ""


@dataclass
class AggregateMetrics:
    """Aggregated stats across multiple runs."""

    api_type: str
    prompt_label: str
    streaming: bool
    runs: list[SingleRunMetrics] = field(default_factory=list)

    # --- computed properties ------------------------------------------------

    @property
    def successful_runs(self) -> list[SingleRunMetrics]:
        return [r for r in self.runs if r.success]

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.runs if not r.success)

    @property
    def error_rate(self) -> float:
        return self.error_count / max(len(self.runs), 1)

    def _stat(self, attr: str) -> dict:
        vals = [getattr(r, attr) for r in self.successful_runs if getattr(r, attr) is not None]
        if not vals:
            return {"min": 0, "max": 0, "mean": 0, "median": 0, "p90": 0, "p99": 0, "stdev": 0}
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        return {
            "min": vals_sorted[0],
            "max": vals_sorted[-1],
            "mean": statistics.mean(vals_sorted),
            "median": statistics.median(vals_sorted),
            "p90": vals_sorted[int(n * 0.9)] if n >= 2 else vals_sorted[-1],
            "p99": vals_sorted[int(n * 0.99)] if n >= 2 else vals_sorted[-1],
            "stdev": statistics.stdev(vals_sorted) if n >= 2 else 0.0,
        }

    @property
    def ttft_stats(self) -> dict:
        return self._stat("time_to_first_token")

    @property
    def total_time_stats(self) -> dict:
        return self._stat("total_time")

    @property
    def tps_stats(self) -> dict:
        return self._stat("tokens_per_second")

    @property
    def latency_stats(self) -> dict:
        return self._stat("end_to_end_latency")

    @property
    def total_cached_tokens(self) -> int:
        return sum(r.cached_tokens for r in self.successful_runs)

    @property
    def cache_hit_rate(self) -> float:
        hits = sum(1 for r in self.successful_runs if r.is_cache_hit)
        return hits / max(len(self.successful_runs), 1)

    @property
    def avg_input_tokens(self) -> float:
        vals = [r.input_tokens for r in self.successful_runs]
        return statistics.mean(vals) if vals else 0

    @property
    def avg_output_tokens(self) -> float:
        vals = [r.output_tokens for r in self.successful_runs]
        return statistics.mean(vals) if vals else 0

    def summary_dict(self) -> dict:
        return {
            "api_type": self.api_type,
            "prompt_label": self.prompt_label,
            "streaming": self.streaming,
            "runs": len(self.runs),
            "errors": self.error_count,
            "error_rate": round(self.error_rate, 4),
            "avg_input_tokens": round(self.avg_input_tokens, 1),
            "avg_output_tokens": round(self.avg_output_tokens, 1),
            "ttft": {k: round(v, 4) for k, v in self.ttft_stats.items()},
            "total_time": {k: round(v, 4) for k, v in self.total_time_stats.items()},
            "tokens_per_second": {k: round(v, 2) for k, v in self.tps_stats.items()},
            "latency": {k: round(v, 4) for k, v in self.latency_stats.items()},
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "total_cached_tokens": self.total_cached_tokens,
        }
