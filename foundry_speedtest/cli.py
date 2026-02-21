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

import math
import statistics
import time
from datetime import datetime, timezone

import fire
from rich.align import Align
from rich.console import Console, Group
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
    TimeRemainingColumn,
)
from rich.rule import Rule
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
        "bar.back": "grey23",
        "bar.complete": "bright_green",
        "bar.finished": "bold bright_green",
        "bar.pulse": "green",
        "progress.description": "bright_green",
        "progress.percentage": "bold bright_green",
        "progress.remaining": "green",
    }
)

console = Console(theme=MATRIX_THEME)

MATRIX_BORDER = Style(color="green")
MATRIX_TITLE = Style(color="bright_green", bold=True)
DIM_BORDER = Style(color="grey37")

_BANNER_LINES = [
    r" ██████╗ ██████╗ ██╗   ██╗███╗   ██╗██████╗ ██████╗ ██╗   ██╗",
    r" ██╔═══╝ ██╔══██╗██║   ██║████╗  ██║██╔══██╗██╔══██╗╚██╗ ██╔╝",
    r" █████╗  ██║  ██║██║   ██║██╔██╗ ██║██║  ██║██████╔╝ ╚████╔╝ ",
    r" ██╔══╝  ██║  ██║██║   ██║██║╚██╗██║██║  ██║██╔══██╗  ╚██╔╝  ",
    r" ██║     ██████╔╝╚██████╔╝██║ ╚████║██████╔╝██║  ██║   ██║   ",
    r" ╚═╝     ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝   ╚═╝   ",
]
_SUBTITLE = "  ░▒▓ S P E E D T E S T  ·  A z u r e  A I  F o u n d r y ▓▒░"

# Colour palette for the wave animation (dark → highlight → dark)
_WAVE_COLORS = [
    "grey23",        # 0 — nearly invisible
    "green4",        # 1
    "green",         # 2
    "bright_green",  # 3
    "bold bright_white",  # 4 — peak highlight
    "bright_green",  # 3
    "green",         # 2
    "green4",        # 1
]
_WAVE_WIDTH = len(_WAVE_COLORS)


def _wave_frame(lines: list[str], tick: int) -> Text:
    """Build one frame of the colour-wave animation across banner characters."""
    max_cols = max(len(ln) for ln in lines)
    frame = Text()
    for row_idx, line in enumerate(lines):
        for col_idx, ch in enumerate(line):
            # Wave position: diagonal sweep (left-to-right + slight row offset)
            dist = (col_idx - tick + row_idx * 2) % (max_cols + _WAVE_WIDTH * 2)
            if 0 <= dist < _WAVE_WIDTH:
                colour = _WAVE_COLORS[dist]
            else:
                colour = "green"  # default body colour
            # Spaces stay unstyled to avoid background flicker
            if ch == " ":
                frame.append(ch)
            else:
                frame.append(ch, style=colour)
        frame.append("\n")
    return frame


def _show_banner():
    """Animated banner: a bright pulse sweeps across the FOUNDRY text."""
    max_cols = max(len(ln) for ln in _BANNER_LINES)
    total_sweep = max_cols + _WAVE_WIDTH * 3  # full width pass
    frames = 28  # number of animation frames
    frame_delay = 0.035  # seconds between frames

    with Live(console=console, refresh_per_second=60, transient=True) as live:
        for f in range(frames):
            tick = int(f * total_sweep / frames)
            banner_text = _wave_frame(_BANNER_LINES, tick)
            # Subtitle pulses in sync — bright at peak, dim otherwise
            sub_brightness = 0.5 + 0.5 * math.sin(f * math.pi / frames)
            if sub_brightness > 0.8:
                sub_style = "bold bright_white"
            elif sub_brightness > 0.5:
                sub_style = "bold bright_green"
            else:
                sub_style = "green"
            subtitle = Text(_SUBTITLE, style=sub_style)

            content = Group(banner_text, subtitle)
            live.update(content)
            time.sleep(frame_delay)

    # Print the final static banner so it stays on screen
    console.print(Text.from_markup(
        "[bright_green]"
        + "\n".join(_BANNER_LINES)
        + "[/bright_green]"
    ))
    console.print(Text(_SUBTITLE, style="bold bright_green"))
    console.print()


# ---------------------------------------------------------------------------
# Formatting helpers
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


def _color_delta(val: float, unit: str = "ms", lower_better: bool = True) -> str:
    """Color a delta value — green for improvement, red for regression."""
    if val == 0:
        return "[dim]—[/dim]"
    sign = "+" if val > 0 else ""
    if unit == "ms":
        text = f"{sign}{val * 1000:.0f}ms"
    else:
        text = f"{sign}{val:.1f}"
    if lower_better:
        return f"[good]{text}[/good]" if val < 0 else f"[error]{text}[/error]"
    else:
        return f"[good]{text}[/good]" if val > 0 else f"[error]{text}[/error]"


def _api_style(api_type: str) -> str:
    return f"api.{api_type}"


def _api_tag(api_type: str) -> str:
    return f"[{_api_style(api_type)}]{api_type}[/{_api_style(api_type)}]"


# ---------------------------------------------------------------------------
# Live dashboard components
# ---------------------------------------------------------------------------


def _build_progress_panel(
    progress: Progress,
    current_phase: str,
    completed: int,
    total: int,
    start_time: float,
) -> Panel:
    """Top panel: big visible progress with phase info."""
    elapsed = time.perf_counter() - start_time
    pct = (completed / total * 100) if total else 0
    eta = ((elapsed / completed) * (total - completed)) if completed > 0 else 0

    header = Text()
    header.append("  ◈ ", style="bright_green")
    header.append(current_phase, style="bold bright_green")
    header.append(f"    {completed}/{total}", style="bright_white")
    header.append(f"  ({pct:.0f}%)", style="bright_green")
    header.append(f"    ⏱ {elapsed:.0f}s elapsed", style="green")
    if completed > 0 and completed < total:
        header.append(f"  · ~{eta:.0f}s remaining", style="dim green")

    content = Group(header, progress)

    return Panel(
        content,
        title="[bold bright_green]▶ PROGRESS[/bold bright_green]",
        border_style=MATRIX_BORDER,
        padding=(0, 1),
    )


def _build_live_results_table(runs: list[SingleRunMetrics], max_rows: int = 12) -> Panel:
    """Middle-left: scrolling live results as runs complete."""
    table = Table(
        show_lines=False,
        border_style=MATRIX_BORDER,
        padding=(0, 1),
        expand=True,
        row_styles=["", "on grey7"],
    )
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("API", width=11)
    table.add_column("Test", style="dim", min_width=14, max_width=22)
    table.add_column("Mode", justify="center", width=6)
    table.add_column("TTFT", justify="right", width=8)
    table.add_column("Total", justify="right", width=8)
    table.add_column("TPS", justify="right", width=7)
    table.add_column("Out", justify="right", width=5)
    table.add_column("Cache", justify="right", width=5)
    table.add_column("", justify="center", width=2)

    # Show most recent runs (tail)
    display_runs = runs[-max_rows:] if len(runs) > max_rows else runs
    start_idx = max(len(runs) - max_rows, 0) + 1

    for i, r in enumerate(display_runs, start_idx):
        status = "[good]✓[/good]" if r.success else "[error]✗[/error]"
        mode = "[bright_green]⇣[/bright_green]" if r.streaming else "[dim]●[/dim]"
        table.add_row(
            str(i),
            _api_tag(r.api_type),
            r.prompt_label[:22],
            mode,
            _color_ttft(r.time_to_first_token),
            _fmt_ms(r.total_time),
            _color_tps(r.tokens_per_second),
            str(r.output_tokens),
            str(r.cached_tokens) if r.cached_tokens else "[dim]·[/dim]",
            status,
        )

    if not runs:
        table.add_row(*["[dim]…[/dim]"] * 10)

    return Panel(
        table,
        title=f"[bold bright_green]📊 LIVE RESULTS[/bold bright_green] [dim]({len(runs)} runs)[/dim]",
        border_style=MATRIX_BORDER,
        padding=(0, 0),
    )


def _build_cold_start_panel(all_runs: list[SingleRunMetrics]) -> Panel:
    """Middle-right: cold start detection and cache analysis."""
    table = Table(
        show_lines=False,
        border_style=DIM_BORDER,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("Indicator", style="info", min_width=14)
    table.add_column("Value", justify="right", style="metric", min_width=10)

    successful = [r for r in all_runs if r.success]

    if not successful:
        table.add_row("[dim]Waiting for data…[/dim]", "")
        return Panel(table, title="[bold bright_green]🧊 COLD START[/bold bright_green]", border_style=MATRIX_BORDER, padding=(0, 0))

    # --- Cold start detection ---
    streaming_runs = [r for r in successful if r.streaming and r.time_to_first_token is not None]

    if len(streaming_runs) >= 2:
        first_ttft = streaming_runs[0].time_to_first_token
        rest_ttfts = [r.time_to_first_token for r in streaming_runs[1:]]
        avg_rest_ttft = statistics.mean(rest_ttfts)
        cold_penalty = first_ttft - avg_rest_ttft

        if cold_penalty > 0.1:
            indicator = "[error]▓▓▓ COLD[/error]"
        elif cold_penalty > 0.03:
            indicator = "[warning]▓▓░ WARM[/warning]"
        else:
            indicator = "[good]▓░░ HOT[/good]"

        table.add_row("Cold Start", indicator)
        table.add_row("1st TTFT", _color_ttft(first_ttft))
        table.add_row("Avg TTFT (rest)", _color_ttft(avg_rest_ttft))
        table.add_row("Cold Penalty", _color_delta(cold_penalty, "ms", lower_better=True))
    else:
        table.add_row("Cold Start", "[dim]need ≥2 stream runs[/dim]")

    table.add_row("", "")

    # --- Cache stats ---
    cached_runs = [r for r in successful if r.cached_tokens > 0]
    total_cached = sum(r.cached_tokens for r in successful)
    cache_rate = len(cached_runs) / len(successful) if successful else 0

    if total_cached > 0:
        cache_indicator = f"[good]{cache_rate * 100:.0f}%[/good]" if cache_rate > 0.3 else f"[warning]{cache_rate * 100:.0f}%[/warning]"
    else:
        cache_indicator = "[dim]0%[/dim]"

    table.add_row("Cache Hit Rate", cache_indicator)
    table.add_row("Cached Tokens", f"[metric]{total_cached}[/metric]")
    table.add_row("Cache Hits", f"{len(cached_runs)}/{len(successful)}")

    table.add_row("", "")

    # --- Running aggregate stats ---
    all_ttfts = [r.time_to_first_token for r in streaming_runs] if streaming_runs else []
    all_tps = [r.tokens_per_second for r in successful if r.tokens_per_second > 0]
    all_totals = [r.total_time for r in successful]
    errors = sum(1 for r in all_runs if not r.success)

    if all_ttfts:
        table.add_row("TTFT P50", _fmt_ms(sorted(all_ttfts)[len(all_ttfts) // 2]))
        if len(all_ttfts) >= 5:
            table.add_row("TTFT P90", _fmt_ms(sorted(all_ttfts)[int(len(all_ttfts) * 0.9)]))

    if all_tps:
        table.add_row("TPS Mean", _color_tps(statistics.mean(all_tps)))
        if len(all_tps) >= 2:
            table.add_row("TPS StdDev", f"[dim]±{statistics.stdev(all_tps):.1f}[/dim]")

    if all_totals:
        table.add_row("Avg Total", _fmt_ms(statistics.mean(all_totals)))

    err_str = f"[error]{errors}[/error]" if errors else "[good]0[/good]"
    table.add_row("Errors", err_str)

    return Panel(
        table,
        title="[bold bright_green]🧊 COLD START · CACHE[/bold bright_green]",
        border_style=MATRIX_BORDER,
        padding=(0, 0),
    )


def _build_phase_log(phases: list[str]) -> Panel:
    """Bottom: completed phase log with checkmarks."""
    lines = []
    for phase in phases:
        lines.append(Text.from_markup(f"  [good]✓[/good] {phase}"))
    if not lines:
        lines.append(Text.from_markup("  [dim]Starting…[/dim]"))

    content = Group(*lines[-6:])
    return Panel(
        content,
        title="[bold bright_green]✓ COMPLETED PHASES[/bold bright_green]",
        border_style=DIM_BORDER,
        padding=(0, 1),
        height=min(len(lines) + 2, 8),
    )


def _build_live_layout(
    progress: Progress,
    current_phase: str,
    completed: int,
    total: int,
    start_time: float,
    all_runs: list[SingleRunMetrics],
    completed_phases: list[str],
) -> Layout:
    """Assemble the full split-panel live dashboard."""
    layout = Layout()

    layout.split_column(
        Layout(name="progress", size=5),
        Layout(name="body", ratio=1),
        Layout(name="phases", size=min(len(completed_phases) + 3, 9)),
    )

    layout["body"].split_row(
        Layout(name="results", ratio=3),
        Layout(name="sidebar", ratio=1, minimum_size=30),
    )

    layout["progress"].update(
        _build_progress_panel(progress, current_phase, completed, total, start_time)
    )
    layout["results"].update(_build_live_results_table(all_runs))
    layout["sidebar"].update(_build_cold_start_panel(all_runs))
    layout["phases"].update(_build_phase_log(completed_phases))

    return layout


# ---------------------------------------------------------------------------
# Static result tables (printed after Live ends)
# ---------------------------------------------------------------------------


def _build_run_table(runs: list[SingleRunMetrics], title: str) -> Table:
    """Full table showing all individual run results."""
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
        status = "[good]✓[/good]" if r.success else "[error]✗[/error]"
        table.add_row(
            str(i),
            _api_tag(r.api_type),
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
    """Aggregate stats comparison table."""
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
            _api_tag(a.api_type),
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


def _build_cold_start_summary(all_runs: list[SingleRunMetrics]) -> Panel:
    """Final cold-start analysis with per-API breakdown."""
    table = Table(
        title="🧊 COLD START ANALYSIS",
        title_style=MATRIX_TITLE,
        border_style=MATRIX_BORDER,
        show_lines=True,
        padding=(0, 1),
    )
    table.add_column("API", style="info", width=14)
    table.add_column("1st TTFT", justify="right", width=10)
    table.add_column("Avg TTFT\n(rest)", justify="right", width=10)
    table.add_column("Cold\nPenalty", justify="right", width=10)
    table.add_column("1st Total", justify="right", width=10)
    table.add_column("Avg Total\n(rest)", justify="right", width=10)
    table.add_column("Verdict", justify="center", width=16)

    for api_type in ("completions", "responses"):
        streaming = [
            r for r in all_runs
            if r.success and r.api_type == api_type and r.streaming and r.time_to_first_token is not None
        ]
        all_api = [r for r in all_runs if r.success and r.api_type == api_type]

        if len(streaming) < 2:
            table.add_row(_api_tag(api_type), *["[dim]—[/dim]"] * 5, "[dim]insufficient data[/dim]")
            continue

        first_ttft = streaming[0].time_to_first_token
        rest_ttfts = [r.time_to_first_token for r in streaming[1:]]
        avg_rest = statistics.mean(rest_ttfts)
        penalty = first_ttft - avg_rest

        first_total = all_api[0].total_time if all_api else 0
        rest_totals = [r.total_time for r in all_api[1:]] if len(all_api) >= 2 else []
        avg_total_rest = statistics.mean(rest_totals) if rest_totals else 0

        if penalty > 0.1:
            verdict = "[error]▓▓▓ COLD START[/error]"
        elif penalty > 0.03:
            verdict = "[warning]▓▓░ WARMING UP[/warning]"
        else:
            verdict = "[good]▓░░ HOT[/good]"

        table.add_row(
            _api_tag(api_type),
            _color_ttft(first_ttft),
            _color_ttft(avg_rest),
            _color_delta(penalty, "ms"),
            _fmt_ms(first_total),
            _fmt_ms(avg_total_rest),
            verdict,
        )

    return Panel(table, border_style=MATRIX_BORDER)


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

    c_cache = sum(a.cache_hit_rate for a in completions_aggs) / max(len(completions_aggs), 1)
    r_cache = sum(a.cache_hit_rate for a in responses_aggs) / max(len(responses_aggs), 1)
    cache_winner = "[api.completions]Completions[/api.completions]" if c_cache >= r_cache else "[api.responses]Responses[/api.responses]"
    table.add_row("Cache Hit Rate", f"{c_cache*100:.0f}%", f"{r_cache*100:.0f}%", cache_winner)

    c_err = sum(a.error_rate for a in completions_aggs) / max(len(completions_aggs), 1)
    r_err = sum(a.error_rate for a in responses_aggs) / max(len(responses_aggs), 1)
    err_winner = "[api.completions]Completions[/api.completions]" if c_err <= r_err else "[api.responses]Responses[/api.responses]"
    table.add_row("Error Rate", f"{c_err*100:.1f}%", f"{r_err*100:.1f}%", err_winner)

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

        # Count total work items
        prompt_tests = len(cfg.prompt_keys) * len(api_list) * 2  # stream + non-stream
        cache_tests = len(api_list)
        concurrency_tests = len(api_list)
        total_tasks = prompt_tests + cache_tests + concurrency_tests

        all_aggregates: list[AggregateMetrics] = []
        all_runs: list[SingleRunMetrics] = []
        completed_phases: list[str] = []
        completed_count = 0
        current_phase = "Initializing…"
        start_time = time.perf_counter()

        # Config summary
        config_table = Table(show_lines=False, border_style=MATRIX_BORDER, padding=(0, 1), expand=True)
        config_table.add_column("Parameter", style="info")
        config_table.add_column("Value", style="metric")
        for k, v in [
            ("Model", model), ("APIs", apis), ("Iterations", str(iterations)),
            ("Warmup", str(warmup)), ("Max Tokens", str(max_tokens)),
            ("Temperature", str(temperature)), ("Concurrency", str(concurrency)),
            ("Cache Rounds", str(cache_rounds)), ("Prompts", ", ".join(cfg.prompt_keys)),
        ]:
            config_table.add_row(k, v)
        console.print(Panel(
            config_table,
            title="[bold bright_green]⚙ CONFIGURATION[/bold bright_green]",
            border_style=MATRIX_BORDER,
        ))
        console.print()

        # --- Progress bar (embedded in the layout) ---
        progress = Progress(
            SpinnerColumn("dots12", style="bright_green"),
            TextColumn("[bold bright_green]{task.description}[/bold bright_green]"),
            BarColumn(
                bar_width=None,
                style="grey23",
                complete_style="bright_green",
                finished_style="bold bright_green",
                pulse_style="green",
            ),
            TextColumn("[bright_white]{task.percentage:>3.0f}%[/bright_white]"),
            MofNCompleteColumn(),
            TextColumn("[dim green]⏱[/dim green]"),
            TimeElapsedColumn(),
            TextColumn("[dim green]→[/dim green]"),
            TimeRemainingColumn(),
            expand=True,
        )
        master_task = progress.add_task("Starting…", total=total_tasks)

        # --- Live dashboard ---
        with Live(console=console, refresh_per_second=6, screen=False) as live:

            def _refresh():
                progress.update(master_task, description=current_phase, completed=completed_count)
                live.update(
                    _build_live_layout(
                        progress, current_phase, completed_count, total_tasks,
                        start_time, all_runs, completed_phases,
                    )
                )

            _refresh()

            # 1) Per-prompt benchmarks
            for api_type in api_list:
                for prompt_key in cfg.prompt_keys:
                    label = BENCHMARK_PROMPTS[prompt_key]["label"]
                    for stream in (True, False):
                        mode = "⇣stream" if stream else "●sync"
                        current_phase = f"{api_type} · {label} ({mode})"
                        _refresh()

                        agg = benchmark_prompt(api_type, prompt_key, cfg, stream=stream)
                        all_aggregates.append(agg)
                        all_runs.extend(agg.runs)
                        completed_count += 1
                        completed_phases.append(f"{_api_tag(api_type)} {label} ({mode})")
                        _refresh()

            # 2) Cache tests
            for api_type in api_list:
                current_phase = f"{api_type} · Cache warm/cold"
                _refresh()

                agg = benchmark_cache(api_type, cfg)
                all_aggregates.append(agg)
                all_runs.extend(agg.runs)
                completed_count += 1
                completed_phases.append(f"{_api_tag(api_type)} Cache warm/cold")
                _refresh()

            # 3) Concurrency tests
            for api_type in api_list:
                current_phase = f"{api_type} · Concurrency x{concurrency}"
                _refresh()

                agg = benchmark_concurrency(api_type, cfg)
                all_aggregates.append(agg)
                all_runs.extend(agg.runs)
                completed_count += 1
                completed_phases.append(f"{_api_tag(api_type)} Concurrency x{concurrency}")
                _refresh()

            current_phase = "✓ All benchmarks complete"
            progress.update(master_task, description=current_phase, completed=total_tasks)
            _refresh()
            time.sleep(0.5)

        # --- Final static results ---
        console.print()
        console.print(Rule("[bold bright_green]FINAL RESULTS[/bold bright_green]", style="bright_green"))
        console.print()

        console.print(_build_run_table(all_runs, f"📊 ALL RUNS — {model}"))
        console.print()

        console.print(_build_summary_table(all_aggregates, f"📈 AGGREGATE STATISTICS — {model}"))
        console.print()

        console.print(_build_cold_start_summary(all_runs))
        console.print()

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

        for mdl in (model1, model2):
            console.print(_build_summary_table(all_aggs[mdl], f"📈 {mdl}"))
            console.print()

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
