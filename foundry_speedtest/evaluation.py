"""Quality evaluation via Azure AI Evaluation library.

This module is *opt-in* — the dependency lives behind the ``eval`` extra
(``pip install -e .[eval]``). When evaluation is enabled, each captured
``response_text`` is graded by one or more LLM-judge evaluators and the
scores are attached to the run metrics.

Default evaluators: ``relevance``, ``coherence``, ``fluency`` — they only
need ``query`` + ``response`` so they work with the existing speed-test
prompts (no ground truth required).

The judge LLM is configured via ``EvaluationConfig`` (env-driven by
default — see ``EvaluationConfig.from_env``). API-key auth is preferred
when ``judge_api_key`` is set; otherwise we fall back to
``DefaultAzureCredential``.
"""
from __future__ import annotations

import concurrent.futures
import os
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit


# Per Azure AI Evaluation docs: each evaluator returns its score under a
# canonical key (e.g. RelevanceEvaluator -> "relevance"). The "_reason"
# key carries the rationale; the "_threshold" key carries the configured
# pass/fail threshold; "_result" carries the binarised label.
DEFAULT_EVALUATORS: tuple[str, ...] = ("relevance", "coherence", "fluency")
SUPPORTED_EVALUATORS: dict[str, str] = {
    "relevance": "RelevanceEvaluator",
    "coherence": "CoherenceEvaluator",
    "fluency": "FluencyEvaluator",
    "similarity": "SimilarityEvaluator",   # needs ground_truth
    "f1_score": "F1ScoreEvaluator",        # NLP, needs ground_truth
}

# Evaluators that need ``ground_truth`` in addition to query + response.
GROUND_TRUTH_EVALUATORS: frozenset[str] = frozenset({"similarity", "f1_score"})

# AI-assisted evaluators that require an LLM judge (model_config).
LLM_JUDGE_EVALUATORS: frozenset[str] = frozenset(
    {"relevance", "coherence", "fluency", "similarity"}
)


@dataclass
class EvaluatorOutcome:
    """Single evaluator result for a single run."""
    name: str
    score: float | None = None
    reason: str = ""
    threshold: float | None = None
    passed: bool | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.error == "" and self.score is not None


@dataclass
class EvaluationConfig:
    """Runtime configuration for the LLM-judge evaluation phase."""
    judge_model: str = "gpt-4.1"
    judge_endpoint: str | None = None
    judge_api_key: str | None = None
    judge_api_version: str = "2024-10-21"
    evaluators: tuple[str, ...] = DEFAULT_EVALUATORS
    max_workers: int = 4
    threshold: int = 3

    @staticmethod
    def from_env(
        *,
        judge_model: str | None = None,
        judge_endpoint: str | None = None,
        judge_api_key: str | None = None,
        evaluators: tuple[str, ...] | None = None,
        max_workers: int | None = None,
    ) -> "EvaluationConfig":
        # Load .env so users don't have to pre-populate the shell.
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        cfg = EvaluationConfig(
            judge_model=judge_model
                or os.getenv("FOUNDRY_SPEEDTEST_JUDGE_MODEL")
                or "gpt-4.1",
            judge_endpoint=judge_endpoint
                or os.getenv("FOUNDRY_SPEEDTEST_JUDGE_ENDPOINT")
                or os.getenv("AZURE_FOUNDRY_JUDGE_ENDPOINT")
                or os.getenv("AZURE_FOUNDRY_ENDPOINT"),
            judge_api_key=judge_api_key
                or os.getenv("FOUNDRY_SPEEDTEST_JUDGE_KEY")
                or os.getenv("AZURE_FOUNDRY_JUDGE_KEY"),
            judge_api_version=os.getenv(
                "FOUNDRY_SPEEDTEST_JUDGE_API_VERSION", "2024-10-21"
            ),
            evaluators=evaluators or DEFAULT_EVALUATORS,
            max_workers=max_workers or int(os.getenv("FOUNDRY_SPEEDTEST_EVAL_WORKERS", "4")),
        )
        return cfg

    def normalised_endpoint(self) -> str | None:
        """Return the resource-root endpoint expected by azure-ai-evaluation.

        The library appends ``/openai/deployments/{deployment}/...`` itself,
        so we strip any trailing ``/openai`` or ``/openai/v1`` suffixes
        introduced for the OpenAI v1 client.
        """
        if not self.judge_endpoint:
            return None
        ep = self.judge_endpoint.strip().rstrip("/")
        parsed = urlsplit(ep)
        path = parsed.path.rstrip("/")
        lower = path.lower()
        if lower.endswith("/openai/v1"):
            path = path[: -len("/openai/v1")]
        elif lower.endswith("/openai"):
            path = path[: -len("/openai")]
        return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))

    def to_model_config(self) -> dict[str, Any]:
        """Build the ``model_config`` dict consumed by Azure AI Evaluation."""
        endpoint = self.normalised_endpoint()
        if not endpoint:
            raise EnvironmentError(
                "Judge endpoint not set. Provide --judge-endpoint, set "
                "FOUNDRY_SPEEDTEST_JUDGE_ENDPOINT, or fall back to "
                "AZURE_FOUNDRY_ENDPOINT."
            )
        cfg: dict[str, Any] = {
            "azure_endpoint": endpoint,
            "azure_deployment": self.judge_model,
            "api_version": self.judge_api_version,
        }
        if self.judge_api_key:
            cfg["api_key"] = self.judge_api_key
        return cfg


class EvaluatorRunner:
    """Lazy-loads selected evaluators and runs them against (query, response).

    Evaluators are instantiated on first use so importing this module never
    triggers the heavy ``azure-ai-evaluation`` import path. Failures during
    instantiation surface immediately on first call.
    """

    def __init__(self, cfg: EvaluationConfig) -> None:
        self.cfg = cfg
        self._evaluators: dict[str, Callable] = {}
        self._init_error: str | None = None

    def _resolve_class(self, name: str):
        try:
            from azure.ai import evaluation as az_eval
        except ImportError as exc:
            raise ImportError(
                "azure-ai-evaluation is not installed. Install with: "
                "pip install -e .[eval]"
            ) from exc
        cls_name = SUPPORTED_EVALUATORS.get(name)
        if not cls_name:
            raise ValueError(
                f"Unknown evaluator {name!r}. Supported: {', '.join(SUPPORTED_EVALUATORS)}"
            )
        cls = getattr(az_eval, cls_name, None)
        if cls is None:
            raise ImportError(
                f"{cls_name} not available in installed azure-ai-evaluation version"
            )
        return cls

    def _build(self, name: str):
        cls = self._resolve_class(name)
        kwargs: dict[str, Any] = {"threshold": self.cfg.threshold}
        if name in LLM_JUDGE_EVALUATORS:
            kwargs["model_config"] = self.cfg.to_model_config()
        return cls(**kwargs)

    def _get(self, name: str):
        if name not in self._evaluators:
            self._evaluators[name] = self._build(name)
        return self._evaluators[name]

    def evaluate(
        self,
        query: str,
        response: str,
        *,
        ground_truth: str | None = None,
        context: str | None = None,
    ) -> dict[str, EvaluatorOutcome]:
        """Run all configured evaluators on a single (query, response) pair."""
        results: dict[str, EvaluatorOutcome] = {}
        if not response:
            return {
                name: EvaluatorOutcome(name=name, error="empty response")
                for name in self.cfg.evaluators
            }

        for name in self.cfg.evaluators:
            outcome = EvaluatorOutcome(name=name)
            if name in GROUND_TRUTH_EVALUATORS and not ground_truth:
                outcome.error = "ground_truth not provided"
                results[name] = outcome
                continue

            try:
                evaluator = self._get(name)
                call_kwargs: dict[str, Any] = {"query": query, "response": response}
                if name in GROUND_TRUTH_EVALUATORS:
                    call_kwargs["ground_truth"] = ground_truth
                if context is not None:
                    call_kwargs["context"] = context
                raw = evaluator(**call_kwargs)
            except Exception as exc:
                outcome.error = f"{type(exc).__name__}: {exc}"
                results[name] = outcome
                continue

            outcome.score = _coerce_score(raw, name)
            outcome.reason = _coerce_reason(raw, name)
            outcome.threshold = _coerce_threshold(raw, name)
            outcome.passed = _coerce_passed(raw, name)
            results[name] = outcome

        return results

    def evaluate_many(
        self,
        items: list[tuple[Any, str, str]],
        *,
        on_done: Callable[[Any, dict[str, EvaluatorOutcome]], None] | None = None,
    ) -> dict[Any, dict[str, EvaluatorOutcome]]:
        """Evaluate many (key, query, response) tuples in parallel.

        Returns ``{key: {evaluator_name: outcome}}``.
        """
        out: dict[Any, dict[str, EvaluatorOutcome]] = {}
        if not items:
            return out

        max_workers = max(1, min(self.cfg.max_workers, len(items)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(self.evaluate, query, response): key
                for key, query, response in items
            }
            for fut in concurrent.futures.as_completed(future_map):
                key = future_map[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    res = {
                        name: EvaluatorOutcome(name=name, error=f"{type(exc).__name__}: {exc}")
                        for name in self.cfg.evaluators
                    }
                out[key] = res
                if on_done:
                    on_done(key, res)
        return out


def _coerce_score(raw: Any, name: str) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        for key in (name, f"gpt_{name}", f"{name}_score"):
            if key in raw and raw[key] is not None:
                try:
                    return float(raw[key])
                except (TypeError, ValueError):
                    continue
        # fallback: first numeric value
        for v in raw.values():
            if isinstance(v, (int, float)):
                return float(v)
    elif isinstance(raw, (int, float)):
        return float(raw)
    return None


def _coerce_reason(raw: Any, name: str) -> str:
    if isinstance(raw, dict):
        for key in (f"{name}_reason", f"gpt_{name}_reason", "reason"):
            if key in raw and raw[key]:
                return str(raw[key])
    return ""


def _coerce_threshold(raw: Any, name: str) -> float | None:
    if isinstance(raw, dict):
        for key in (f"{name}_threshold", "threshold"):
            if key in raw and raw[key] is not None:
                try:
                    return float(raw[key])
                except (TypeError, ValueError):
                    continue
    return None


def _coerce_passed(raw: Any, name: str) -> bool | None:
    if isinstance(raw, dict):
        for key in (f"{name}_result", "result", "passed"):
            if key in raw:
                v = raw[key]
                if isinstance(v, bool):
                    return v
                if isinstance(v, str):
                    return v.lower() in {"pass", "passed", "true"}
    return None
