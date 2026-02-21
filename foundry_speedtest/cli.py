"""
Foundry SpeedTest CLI — Matrix-themed terminal UI for LLM benchmarking.

Usage:
    python -m foundry_speedtest.cli bench <model>           # full benchmark suite
    python -m foundry_speedtest.cli bench <model> --apis completions
    python -m foundry_speedtest.cli bench <model> --iterations 5
    python -m foundry_speedtest.cli quick <model>           # fast single-prompt check
    python -m foundry_speedtest.cli compare <model1> <model2>  # head-to-head
"""

from __future__ import annotations

import sys
import time
from typing import Optional

import fire
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from .benchmarks import benchmark_cache, benchmark_concurrency, benchmark_prompt
from .config import BENCHMARK_PROMPTS, BenchmarkConfig
from .metrics import AggregateMetrics, SingleRunMetrics
from .reporter import export_csv, export_json, export_raw_csv

# ---------------------------------------------------------------------------
# Matrix theme
# ---------------------------------------------------------------------------

MATRIX_THEME = Theme(
    {
        "info": "bright_green",
        "warning": "bright_yellow",
        "error": "bold bright_red",
        "header": "bold bright_green on black",
        "metric": "bright_green",
        "dim": "green",
        "highlight": "bold bright_white on green",
        "api.completions": "bright_cyan",
        "api.responses": "bright_magenta",
        "good": "bold bright_green",
        "bad": "bold bright_red",
        "muted": "dim green",
    }
)

console = Console(theme=MATRIX_THEME)

MATRIX_BORDER = Style(color="green")
MATRIX_TITLE = Style(color="bright_green", bold=True)

BANNER = r"""
[bright_green]
 ██████╗ ██████╗ ██╗   ██╗███╗   ██╗██████╗ ██████╗ ██╗   ██╗
 ██╔═══╝ ██╔══██╗██║   ██║████╗  ██║██╔══██╗██╔══██╗╚██╗ ██╔╝
 █████╗  ██║  ██║██║   ██║██╔██╗ ██║██║  ██║██████╔╝ ╚████╔╝
 ██╔══╝  ██║  ██║██║   ██║██║╚██╗██║██║  ██║██╔══██╗  ╚██╔╝
 ██║     ██████╔╝╚██████╔╝██║ ╚████║██████╔╝██║  ██║   ██║
 ╚═╝     ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝   ╚═╝
[/bright_green]
[bold bright_green]  ░▒▓ S P E E D T E S T  ·  A z u r e  A I  F o u n d r y ▓▒░[/bold bright_green]
"""


def _show_banner():
    console.print(BANNER)
    console.print()


# ---------------------------------------------------------------------------
# Live-updating result tables
# ---------------------------------------------------------------------------


def _fmt_ms(val: float | None) -> str:
    if val is None or val == 0:
        return "—"
    return f"{val * 1000:.0f}ms"


def _fmt_tps(val: float) -> str:
    if val == 0:
        return "—"
    return f"{val:.1f}"


def _color_ttft(val: float | None) -> str:
    if val is None or val == 0:
        return "[dim]—[/dim]"
    ms = val * 1000
    if ms < 200:
        return f"[good]{ms:.0f}ms[/good]"
    if ms < 500:
        return f"[warning]{ms:.0f}ms[/warning]"
    return f"[error]{ms:.0f}ms[/error]"


def _color_tps(val: float) -> str:
    if val == 0:
        return "[dim]—[/dim]"
    if val > 80:
        return f"[good]{val:.1f}[/good]"
    if val > 30:
        return f"[info]{val:.1f}[/info]"
    return f"[warning]{val:.1f}[/warning]"


def _api_style(api_type: str) -> str:
    return f"api.{api_type}"


def _build_run_table(runs: list[SingleRunMetrics], title: str) -> Table:
    """Build a table showing individual run results."""
    table = Table(
        title=title,
        title_style=MATRIX_TITLE,
        border_style=MATRIX_BORDER,
        show_lines=True,
        padding=(0, 1),
    )
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("API", style="info", width=12)
    table.add_column("Prompt", style="dim", min_width=16)
    table.add_column("Stream", justify="center", width=6)
    table.add_column("TTFT", justify="right", width=8)
    table.add_column("Total", justify="right", width=8)
    table.add_column("TPS", justify="right", width=8)
    table.add_column("In Tok", justify="right", width=7)
    table.add_column("Out Tok", justify="right", width=7)
    table.add_column("Cached", justify="right", width=7)
    table.add_column("Status", justify="center", width=6)

    for i, r in enumerate(runs, 1):
        status = "[good]✓[/good]" if r.success else f"[error]✗[/error]"
        table.add_row(
            str(i),
            f"[{_api_style(r.api_type)}]{r.api_type}[/{_api_style(r.api_type)}]",
            r.prompt_label[:24],
            "✓" if r.streaming else "—",
            _color_ttft(r.time_to_first_token),
            _fmt_ms(r.total_time),
            _color_tps(r.tokens_per_second),
            str(r.input_tokens),
            str(r.output_tokens),
            str(r.cached_tokens) if r.cached_tokens else "—",
            status,
        )

    return table


def _build_summary_table(aggregates: list[AggregateMetrics], title: str) -> Table:
    """Build the final summary comparison table."""
    table = Table(
        title=title,
        title_style=MATRIX_TITLE,
        border_style=MATRIX_BORDER,
        show_lines=True,
        padding=(0, 1),
    )
    table.add_column("API", style="info", width=12)
    table.add_column("Test", style="dim", min_width=20)
    table.add_column("Stream", justify="center", width=6)
    table.add_column("Runs", justify="right", width=5)
    table.add_column("Err%", justify="right", width=6)
    table.add_column("TTFT\nMean", justify="right", width=8)
    table.add_column("TTFT\nP90", justify="right", width=8)
    table.add_column("Total\nMean", justify="right", width=8)
    table.add_column("Total\nP90", justify="right", width=8)
    table.add_column("TPS\nMean", justify="right", width=8)
    table.add_column("TPS\nP90", justify="right", width=8)
    table.add_column("Avg In", justify="right", width=7)
    table.add_column("Avg Out", justify="right", width=7)
    table.add_column("Cache\nHit%", justify="right", width=7)

    for a in aggregates:
        s = a.summary_dict()
        err_str = f"[error]{s['error_rate']*100:.0f}%[/error]" if s["error_rate"] > 0 else "[good]0%[/good]"
        table.add_row(
            f"[{_api_style(a.api_type)}]{a.api_type}[/{_api_style(a.api_type)}]",
            a.prompt_label,
            "✓" if a.streaming else "—",
            str(s["runs"]),
            err_str,
            _color_ttft(s["ttft"]["mean"]),
            _color_ttft(s["ttft"]["p90"]),
            _fmt_ms(s["total_time"]["mean"]),
            _fmt_ms(s["total_time"]["p90"]),
            _color_tps(s["tokens_per_second"]["mean"]),
            _color_tps(s["tokens_per_second"]["p90"]),
            str(round(s["avg_input_tokens"])),
            str(round(s["avg_output_tokens"])),
            f"{s['cache_hit_rate']*100:.0f}%",
        )

    return table


def _build_comparison_panel(
    completions_aggs: list[AggregateMetrics],
    responses_aggs: list[AggregateMetrics],
) -> Panel:
    """Side-by-side comparison of Completions vs Responses API."""
    table = Table(
        title="⚡ HEAD-TO-HEAD COMPARISON",
        title_style=MATRIX_TITLE,
        border_style=MATRIX_BORDER,
        show_lines=True,
    )
    table.add_column("Metric", style="info", min_width=20)
    table.add_column("Completions API", justify="right", style="api.completions", min_width=16)
    table.add_column("Responses API", justify="right", style="api.responses", min_width=16)
    table.add_column("Winner", justify="center", width=14)

    def _avg_metric(aggs: list[AggregateMetrics], stat_name: str, sub_key: str) -> float:
        vals = [getattr(a, f"{stat_name}_stats")[sub_key] for a in aggs if getattr(a, f"{stat_name}_stats")[sub_key]]
        return sum(vals) / len(vals) if vals else 0

    comparisons = [
        ("TTFT Mean", "ttft", "mean", True),
        ("TTFT P90", "ttft", "p90", True),
        ("Total Time Mean", "total_time", "mean", True),
        ("Total Time P90", "total_time", "p90", True),
        ("TPS Mean", "tps", "mean", False),
        ("TPS P90", "tps", "p90", False),
        ("Latency Mean", "latency", "mean", True),
    ]

    for label, stat, sub, lower_better in comparisons:
        c_val = _avg_metric(completions_aggs, stat, sub)
        r_val = _avg_metric(responses_aggs, stat, sub)

        c_str = _fmt_ms(c_val) if "time" in stat or stat in ("ttft", "latency") else _fmt_tps(c_val)
        r_str = _fmt_ms(r_val) if "time" in stat or stat in ("ttft", "latency") else _fmt_tps(r_val)

        if c_val == 0 and r_val == 0:
            winner = "[dim]—[/dim]"
        elif lower_better:
            winner = "[api.completions]Completions[/api.completions]" if c_val <= r_val else "[api.responses]Responses[/api.responses]"
        else:
            winner = "[api.completions]Completions[/api.completions]" if c_val >= r_val else "[api.responses]Responses[/api.responses]"

        table.add_row(label, c_str, r_str, winner)

    # Cache comparison
    c_cache = sum(a.cache_hit_rate for a in completions_aggs) / max(len(completions_aggs), 1)
    r_cache = sum(a.cache_hit_rate for a in responses_aggs) / max(len(responses_aggs), 1)
    c_cache_str = f"{c_cache*100:.0f}%"
    r_cache_str = f"{r_cache*100:.0f}%"
    cache_winner = "[api.completions]Completions[/api.completions]" if c_cache >= r_cache else "[api.responses]Responses[/api.responses]"
    table.add_row("Cache Hit Rate", c_cache_str, r_cache_str, cache_winner)

    # Error rate
    c_err = sum(a.error_rate for a in completions_aggs) / max(len(completions_aggs), 1)
    r_err = sum(a.error_rate for a in responses_aggs) / max(len(responses_aggs), 1)
    table.add_row(
        "Error Rate",
        f"{c_err*100:.1f}%",
        f"{r_err*100:.1f}%",
        "[api.completions]Completions[/api.completions]" if c_err <= r_err else "[api.responses]Responses[/api.responses]",
    )

    return Panel(table, border_style=MATRIX_BORDER)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


class FoundrySpeedTest:
    """🔥 Foundry SpeedTest — benchmark Azure AI Foundry models at the speed of light."""

    def bench(
        self,
        model: str = "gpt-4.1-nano",
        iterations: int = 3,
        warmup: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.7,
        concurrency: int = 5,
        cache_rounds: int = 5,
        apis: str = "both",
        prompts: str = "all",
        timeout: float = 120.0,
        output: str = "all",
    ):
        """
        Run the full benchmark suite.

        Args:
            model:        Model deployment name (e.g. gpt-4.1-nano, gpt-5.2)
            iterations:   Number of measured runs per prompt/api combo
            warmup:       Warmup runs before measuring (discarded)
            max_tokens:   Max output tokens per request
            temperature:  Sampling temperature
            concurrency:  Concurrent requests for throughput test
            cache_rounds: Number of identical calls for cache test
            apis:         Which APIs to test: 'both', 'completions', or 'responses'
            prompts:      Prompt set: 'all' or comma-separated keys (short,medium,long,code,reasoning)
            timeout:      Request timeout in seconds
            output:       Export format: 'all', 'json', 'csv', or 'none'
        """
        _show_banner()

        cfg = BenchmarkConfig(
            model=model,
            iterations=iterations,
            warmup=warmup,
            max_tokens=max_tokens,
            temperature=temperature,
            concurrency=concurrency,
            cache_rounds=cache_rounds,
            timeout=timeout,
        )

        if prompts != "all":
            cfg.prompt_keys = [k.strip() for k in prompts.split(",")]

        api_list = ["completions", "responses"] if apis == "both" else [apis]

        # Count total work items for progress bar
        prompt_tests = len(cfg.prompt_keys) * len(api_list) * 2  # stream + non-stream
        cache_tests = len(api_list)
        concurrency_tests = len(api_list)
        total_tasks = prompt_tests + cache_tests + concurrency_tests

        all_aggregates: list[AggregateMetrics] = []
        all_runs: list[SingleRunMetrics] = []

        # Config summary panel
        config_table = Table(
            title="⚙ CONFIGURATION",
            title_style=MATRIX_TITLE,
            border_style=MATRIX_BORDER,
            show_lines=False,
        )
        config_table.add_column("Parameter", style="info")
        config_table.add_column("Value", style="metric")
        config_table.add_row("Model", model)
        config_table.add_row("APIs", apis)
        config_table.add_row("Iterations", str(iterations))
        config_table.add_row("Warmup", str(warmup))
        config_table.add_row("Max Tokens", str(max_tokens))
        config_table.add_row("Temperature", str(temperature))
        config_table.add_row("Concurrency", str(concurrency))
        config_table.add_row("Cache Rounds", str(cache_rounds))
        config_table.add_row("Prompts", ", ".join(cfg.prompt_keys))
        console.print(Panel(config_table, border_style=MATRIX_BORDER))
        console.print()

        # --- Main progress loop ---
        progress = Progress(
            SpinnerColumn("dots", style="bright_green"),
            TextColumn("[info]{task.description}"),
            BarColumn(bar_width=40, style="green", complete_style="bright_green"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        )

        with progress:
            master = progress.add_task("Overall", total=total_tasks)

            # 1) Per-prompt benchmarks (streaming + non-streaming)
            for api_type in api_list:
                for prompt_key in cfg.prompt_keys:
                    label = BENCHMARK_PROMPTS[prompt_key]["label"]
                    for stream in (True, False):
                        mode = "stream" if stream else "sync"
                        desc = f"[{_api_style(api_type)}]{api_type}[/{_api_style(api_type)}] · {label} ({mode})"
                        progress.update(master, description=desc)

                        def on_run(i, m, _desc=desc):
                            pass  # progress advances per-task not per-run

                        agg = benchmark_prompt(
                            api_type, prompt_key, cfg,
                            stream=stream,
                            on_run=on_run,
                        )
                        all_aggregates.append(agg)
                        all_runs.extend(agg.runs)
                        progress.advance(master)

            # 2) Cache tests
            for api_type in api_list:
                desc = f"[{_api_style(api_type)}]{api_type}[/{_api_style(api_type)}] · Cache warm/cold"
                progress.update(master, description=desc)
                agg = benchmark_cache(api_type, cfg)
                all_aggregates.append(agg)
                all_runs.extend(agg.runs)
                progress.advance(master)

            # 3) Concurrency tests
            for api_type in api_list:
                desc = f"[{_api_style(api_type)}]{api_type}[/{_api_style(api_type)}] · Concurrency x{concurrency}"
                progress.update(master, description=desc)
                agg = benchmark_concurrency(api_type, cfg)
                all_aggregates.append(agg)
                all_runs.extend(agg.runs)
                progress.advance(master)

        console.print()

        # --- Results ---
        console.print(_build_run_table(all_runs, f"📊 ALL RUNS — {model}"))
        console.print()

        console.print(_build_summary_table(all_aggregates, f"📈 AGGREGATE STATISTICS — {model}"))
        console.print()

        # Head-to-head if both APIs tested
        if apis == "both":
            c_aggs = [a for a in all_aggregates if a.api_type == "completions"]
            r_aggs = [a for a in all_aggregates if a.api_type == "responses"]
            console.print(_build_comparison_panel(c_aggs, r_aggs))
            console.print()

        # --- Export ---
        if output in ("all", "json"):
            p = export_json(all_aggregates, model)
            console.print(f"  [info]JSON report →[/info] [bold]{p}[/bold]")
        if output in ("all", "csv"):
            p = export_csv(all_aggregates, model)
            console.print(f"  [info]CSV summary  →[/info] [bold]{p}[/bold]")
            p = export_raw_csv(all_aggregates, model)
            console.print(f"  [info]CSV raw log  →[/info] [bold]{p}[/bold]")
        console.print()
        console.print("[bold bright_green]░▒▓ Benchmark complete. ▓▒░[/bold bright_green]")

    def quick(
        self,
        model: str = "gpt-4.1-nano",
        max_tokens: int = 256,
        timeout: float = 60.0,
    ):
        """
        Quick single-prompt test to verify connectivity and see basic metrics.

        Args:
            model:      Model deployment name
            max_tokens: Max output tokens
            timeout:    Request timeout in seconds
        """
        _show_banner()
        console.print(f"  [info]Quick test →[/info] [bold]{model}[/bold]")
        console.print()

        cfg = BenchmarkConfig(model=model, iterations=1, warmup=0, max_tokens=max_tokens, timeout=timeout)

        results: list[SingleRunMetrics] = []
        for api_type in ("completions", "responses"):
            for stream in (True, False):
                mode = "stream" if stream else "sync"
                with console.status(
                    f"  [bright_green]Testing {api_type} ({mode})...[/bright_green]",
                    spinner="dots",
                    spinner_style="bright_green",
                ):
                    agg = benchmark_prompt(api_type, "medium", cfg, stream=stream)
                    for r in agg.runs:
                        results.append(r)

        console.print(_build_run_table(results, f"⚡ QUICK TEST — {model}"))
        console.print()
        console.print("[bold bright_green]░▒▓ Done. ▓▒░[/bold bright_green]")

    def compare(
        self,
        model1: str,
        model2: str,
        iterations: int = 3,
        max_tokens: int = 512,
        timeout: float = 120.0,
    ):
        """
        Compare two models head-to-head.

        Args:
            model1:     First model deployment name
            model2:     Second model deployment name
            iterations: Runs per test
            max_tokens: Max output tokens
            timeout:    Request timeout
        """
        _show_banner()
        console.print(f"  [info]Comparing[/info] [bold]{model1}[/bold] [info]vs[/info] [bold]{model2}[/bold]")
        console.print()

        all_aggs: dict[str, list[AggregateMetrics]] = {model1: [], model2: []}

        for mdl in (model1, model2):
            cfg = BenchmarkConfig(
                model=mdl, iterations=iterations, warmup=1,
                max_tokens=max_tokens, timeout=timeout,
                prompt_keys=["short", "medium", "long"],
            )
            with console.status(
                f"  [bright_green]Benchmarking {mdl}...[/bright_green]",
                spinner="dots",
                spinner_style="bright_green",
            ):
                for api_type in ("completions", "responses"):
                    for prompt_key in cfg.prompt_keys:
                        for stream in (True, False):
                            agg = benchmark_prompt(api_type, prompt_key, cfg, stream=stream)
                            all_aggs[mdl].append(agg)

        # Show side-by-side
        for mdl in (model1, model2):
            console.print(_build_summary_table(all_aggs[mdl], f"📈 {mdl}"))
            console.print()

        # Cross-model comparison table
        cmp_table = Table(
            title="⚔ MODEL COMPARISON",
            title_style=MATRIX_TITLE,
            border_style=MATRIX_BORDER,
            show_lines=True,
        )
        cmp_table.add_column("Metric", style="info")
        cmp_table.add_column(model1, justify="right", style="api.completions")
        cmp_table.add_column(model2, justify="right", style="api.responses")
        cmp_table.add_column("Winner", justify="center")

        def _global_avg(aggs, attr, sub):
            vals = [getattr(a, f"{attr}_stats")[sub] for a in aggs if getattr(a, f"{attr}_stats")[sub]]
            return sum(vals) / len(vals) if vals else 0

        for label, attr, sub, lower in [
            ("TTFT Mean", "ttft", "mean", True),
            ("Total Time Mean", "total_time", "mean", True),
            ("TPS Mean", "tps", "mean", False),
        ]:
            v1 = _global_avg(all_aggs[model1], attr, sub)
            v2 = _global_avg(all_aggs[model2], attr, sub)
            fmt = _fmt_ms if attr in ("ttft", "total_time") else _fmt_tps
            if lower:
                w = model1 if v1 <= v2 else model2
            else:
                w = model1 if v1 >= v2 else model2
            cmp_table.add_row(label, fmt(v1), fmt(v2), f"[good]{w}[/good]")

        console.print(Panel(cmp_table, border_style=MATRIX_BORDER))
        console.print()
        console.print("[bold bright_green]░▒▓ Comparison complete. ▓▒░[/bold bright_green]")

    def list_prompts(self):
        """Show available benchmark prompt sets."""
        _show_banner()
        table = Table(
            title="📝 BENCHMARK PROMPTS",
            title_style=MATRIX_TITLE,
            border_style=MATRIX_BORDER,
        )
        table.add_column("Key", style="info")
        table.add_column("Label", style="metric")
        table.add_column("Preview", style="dim", max_width=60)
        for key, p in BENCHMARK_PROMPTS.items():
            table.add_row(key, p["label"], p["user"][:60] + "…" if len(p["user"]) > 60 else p["user"])
        console.print(table)


def main():
    fire.Fire(FoundrySpeedTest)


if __name__ == "__main__":
    main()
