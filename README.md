# ⚡ Foundry SpeedTest

> **Matrix-themed CLI benchmark suite for Azure AI Foundry models**
>
> Compare **Completions API** vs **Responses API** head-to-head with real latency metrics, throughput analysis, and cache behaviour testing — all from your terminal.

```
 ██████╗ ██████╗ ██╗   ██╗███╗   ██╗██████╗ ██████╗ ██╗   ██╗
 ██╔═══╝ ██╔══██╗██║   ██║████╗  ██║██╔══██╗██╔══██╗╚██╗ ██╔╝
 █████╗  ██║  ██║██║   ██║██╔██╗ ██║██║  ██║██████╔╝ ╚████╔╝
 ██╔══╝  ██║  ██║██║   ██║██║╚██╗██║██║  ██║██╔══██╗  ╚██╔╝
 ██║     ██████╔╝╚██████╔╝██║ ╚████║██████╔╝██║  ██║   ██║
 ╚═╝     ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝   ╚═╝
  ░▒▓ S P E E D T E S T  ·  M i c r o s o f t  F o u n d r y ▓▒░
```

---

## 📊 What It Measures

| Metric | Description |
|--------|-------------|
| **TTFT** | Time to First Token — how fast the stream starts (streaming mode) |
| **Total Time** | Wall-clock time for the full response |
| **Tokens/sec (TPS)** | Output throughput — tokens generated per second |
| **Input / Output Tokens** | Token counts reported by the API |
| **Cache Hit / Miss** | Detects prompt caching via `cached_tokens` in usage |
| **Concurrent Throughput** | Parallel request performance under load |
| **P50 / P90 / P99** | Percentile latencies across multiple runs |
| **Error Rate** | Failed requests as a percentage of total |

Tests run against **both** the **Completions API** (`chat.completions.create`) and the **Responses API** (`responses.create`) in streaming and non-streaming modes.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    CLI Layer (Fire + Rich)                    │
│   bench  ·  quick  ·  compare  ·  list_prompts               │
└──────────────┬───────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────┐
│                    Benchmark Engine                           │
│  benchmark_prompt · benchmark_cache · benchmark_concurrency   │
└──────────────┬───────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────┐
│                     API Adapters                              │
│         ┌──────────────┐    ┌──────────────┐                 │
│         │ Completions  │    │  Responses   │                 │
│         │  (stream +   │    │  (stream +   │                 │
│         │   sync)      │    │   sync)      │                 │
│         └──────┬───────┘    └──────┬───────┘                 │
└────────────────┼───────────────────┼─────────────────────────┘
                 │                   │
┌────────────────▼───────────────────▼─────────────────────────┐
│              Azure AI Foundry Endpoint                        │
│         (OpenAI-compatible  ·  Entra ID Auth)                 │
└──────────────────────────────────────────────────────────────┘

        ┌────────────────┐     ┌────────────────┐
        │ SingleRunMetrics│────▶│AggregateMetrics│
        │ (per-call data)│     │ (P50/P90/P99)  │
        └────────────────┘     └───────┬────────┘
                                       │
                         ┌─────────────▼──────────────┐
                         │         Reporter            │
                         │  JSON · CSV · Raw CSV Log   │
                         └─────────────────────────────┘
```

> Full Mermaid diagram: [`docs/architecture.mmd`](docs/architecture.mmd)

---

## 📁 Project Structure

```
Foundry-SpeedTest/
├── foundry_speedtest/          # Main package
│   ├── __init__.py
│   ├── __main__.py             # python -m entry point
│   ├── cli.py                  # Fire CLI + Rich Matrix UI
│   ├── benchmarks.py           # Test orchestrator
│   ├── adapters.py             # Completions & Responses API wrappers
│   ├── metrics.py              # SingleRunMetrics + AggregateMetrics
│   ├── config.py               # Prompts, BenchmarkConfig
│   └── reporter.py             # JSON/CSV export
├── API/                        # Sample standalone scripts
│   ├── completions_connection.py
│   └── responses_connection.py
├── docs/
│   └── architecture.mmd        # Mermaid architecture diagram
├── .env.example                # Template for env vars
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Azure AI Foundry** resource with a deployed model
- **Azure CLI** logged in (`az login`) — used for Entra ID authentication

### 1. Clone & Setup

```bash
git clone https://github.com/<your-username>/Foundry-SpeedTest.git
cd Foundry-SpeedTest

# Create virtual environment
python -m venv .venv

# Activate it
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
```

### 2. Install

```bash
pip install -e .
```

### 3. Configure

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your Foundry endpoint
# AZURE_FOUNDRY_ENDPOINT=https://<your-resource>.openai.azure.com/openai/v1
```

### 4. Authenticate

```bash
# Make sure you're logged in to Azure
az login
```

### 5. Run

```bash
# Quick connectivity check
python -m foundry_speedtest quick gpt-4.1-nano

# Full benchmark suite
python -m foundry_speedtest bench gpt-4.1-nano

# Compare two models head-to-head
python -m foundry_speedtest compare gpt-4.1-nano gpt-4.1-mini
```

---

## 🔥 CLI Commands

### `bench` — Full Benchmark Suite

```bash
python -m foundry_speedtest bench <model> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--iterations` | `3` | Measured runs per prompt/API combo |
| `--warmup` | `1` | Warmup runs (discarded) |
| `--max_tokens` | `512` | Max output tokens per request |
| `--temperature` | `0.7` | Sampling temperature |
| `--concurrency` | `5` | Parallel requests for throughput test |
| `--cache_rounds` | `5` | Identical calls for cache detection |
| `--apis` | `both` | `both`, `completions`, or `responses` |
| `--prompts` | `all` | `all` or comma-separated: `short,medium,long,code,reasoning` |
| `--timeout` | `120` | Request timeout (seconds) |
| `--output` | `all` | Export: `all`, `json`, `csv`, or `none` |

**Examples:**

```bash
# Test only Completions API with 5 iterations
python -m foundry_speedtest bench gpt-5.2 --apis completions --iterations 5

# Short + code prompts only, no file export
python -m foundry_speedtest bench gpt-4.1-nano --prompts short,code --output none

# High concurrency stress test
python -m foundry_speedtest bench gpt-4.1-nano --concurrency 20 --cache_rounds 10
```

### `quick` — Fast Connectivity Check

```bash
python -m foundry_speedtest quick <model>
```

Runs a single medium prompt across both APIs (streaming + sync) — good for verifying your setup works.

### `compare` — Model vs Model

```bash
python -m foundry_speedtest compare <model1> <model2>
```

Side-by-side comparison across short/medium/long prompts with a winner declared per metric.

### `list_prompts` — Show Prompt Catalogue

```bash
python -m foundry_speedtest list_prompts
```

---

## 📂 Output & Reports

Results are saved to the `results/` directory:

| File | Contents |
|------|----------|
| `benchmark_<model>_<timestamp>.json` | Full structured results with nested stats |
| `benchmark_<model>_<timestamp>.csv` | Aggregate summary (one row per test) |
| `benchmark_<model>_<timestamp>_raw.csv` | Every individual API call logged |

---

## 🧪 What Gets Tested

| Test | Streaming | Non-Streaming | Description |
|------|:---------:|:-------------:|-------------|
| **Short prompt** | ✓ | ✓ | Trivial question → measures overhead |
| **Medium prompt** | ✓ | ✓ | Technical explanation → balanced test |
| **Long prompt** | ✓ | ✓ | Tutorial generation → heavy output |
| **Code generation** | ✓ | ✓ | Python function → structured output |
| **Reasoning** | ✓ | ✓ | Multi-step logic → reasoning latency |
| **Cache warm/cold** | ✓ | — | Identical prompt repeated N times |
| **Concurrency** | — | ✓ | Parallel requests for throughput |

---

## 🔐 Authentication

This tool uses **Azure Entra ID** (via `DefaultAzureCredential`) — no API keys stored in code. It picks up credentials from:

1. `az login` session
2. Environment variables (`AZURE_CLIENT_ID`, etc.)
3. Managed Identity (when running in Azure)
4. VS Code Azure account

---

## License

MIT
