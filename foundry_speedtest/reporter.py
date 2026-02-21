"""Reporter — CSV/JSON export and summary generation."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .metrics import AggregateMetrics, SingleRunMetrics


def _results_dir() -> Path:
    d = Path("results")
    d.mkdir(exist_ok=True)
    return d


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def export_json(
    all_aggregates: list[AggregateMetrics],
    model: str,
    filename: str | None = None,
) -> str:
    ts = _timestamp()
    fname = filename or f"benchmark_{model}_{ts}.json"
    path = _results_dir() / fname

    payload = {
        "model": model,
        "timestamp": ts,
        "results": [a.summary_dict() for a in all_aggregates],
    }
    path.write_text(json.dumps(payload, indent=2))
    return str(path)


# ---------------------------------------------------------------------------
# CSV export (flat — one row per aggregate)
# ---------------------------------------------------------------------------

_CSV_HEADERS = [
    "api_type", "prompt_label", "streaming", "runs", "errors", "error_rate",
    "avg_input_tokens", "avg_output_tokens",
    "ttft_mean", "ttft_median", "ttft_p90", "ttft_p99",
    "total_time_mean", "total_time_median", "total_time_p90",
    "tps_mean", "tps_median", "tps_p90",
    "latency_mean", "latency_p90",
    "cache_hit_rate", "total_cached_tokens",
]


def export_csv(
    all_aggregates: list[AggregateMetrics],
    model: str,
    filename: str | None = None,
) -> str:
    ts = _timestamp()
    fname = filename or f"benchmark_{model}_{ts}.csv"
    path = _results_dir() / fname

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_HEADERS)
        writer.writeheader()
        for a in all_aggregates:
            s = a.summary_dict()
            row = {
                "api_type": s["api_type"],
                "prompt_label": s["prompt_label"],
                "streaming": s["streaming"],
                "runs": s["runs"],
                "errors": s["errors"],
                "error_rate": s["error_rate"],
                "avg_input_tokens": s["avg_input_tokens"],
                "avg_output_tokens": s["avg_output_tokens"],
                "ttft_mean": s["ttft"]["mean"],
                "ttft_median": s["ttft"]["median"],
                "ttft_p90": s["ttft"]["p90"],
                "ttft_p99": s["ttft"]["p99"],
                "total_time_mean": s["total_time"]["mean"],
                "total_time_median": s["total_time"]["median"],
                "total_time_p90": s["total_time"]["p90"],
                "tps_mean": s["tokens_per_second"]["mean"],
                "tps_median": s["tokens_per_second"]["median"],
                "tps_p90": s["tokens_per_second"]["p90"],
                "latency_mean": s["latency"]["mean"],
                "latency_p90": s["latency"]["p90"],
                "cache_hit_rate": s["cache_hit_rate"],
                "total_cached_tokens": s["total_cached_tokens"],
            }
            writer.writerow(row)

    return str(path)


# ---------------------------------------------------------------------------
# Raw run log (every single call — for deep analysis)
# ---------------------------------------------------------------------------


def export_raw_csv(
    all_aggregates: list[AggregateMetrics],
    model: str,
    filename: str | None = None,
) -> str:
    ts = _timestamp()
    fname = filename or f"benchmark_{model}_{ts}_raw.csv"
    path = _results_dir() / fname

    headers = [
        "api_type", "prompt_label", "streaming", "success", "error",
        "ttft", "total_time", "e2e_latency",
        "input_tokens", "output_tokens", "total_tokens",
        "tokens_per_second", "cached_tokens", "is_cache_hit",
        "model_id", "finish_reason", "system_fingerprint",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for a in all_aggregates:
            for r in a.runs:
                writer.writerow({
                    "api_type": r.api_type,
                    "prompt_label": r.prompt_label,
                    "streaming": r.streaming,
                    "success": r.success,
                    "error": r.error or "",
                    "ttft": r.time_to_first_token or "",
                    "total_time": round(r.total_time, 6),
                    "e2e_latency": round(r.end_to_end_latency, 6),
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "total_tokens": r.total_tokens,
                    "tokens_per_second": round(r.tokens_per_second, 2),
                    "cached_tokens": r.cached_tokens,
                    "is_cache_hit": r.is_cache_hit,
                    "model_id": r.model_id,
                    "finish_reason": r.finish_reason,
                    "system_fingerprint": r.system_fingerprint,
                })

    return str(path)
