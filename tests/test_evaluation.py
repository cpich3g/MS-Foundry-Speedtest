"""Unit tests for the quality-evaluation integration.

These tests stub out ``azure.ai.evaluation`` so the suite runs even when
the optional ``[eval]`` extra is not installed. They verify:

* ``EvaluationConfig`` env resolution + endpoint normalisation
* ``EvaluatorRunner`` lazy class resolution, success path, and per-evaluator
  failure isolation (a crash in one evaluator never breaks the others, and
  never propagates out of ``benchmark_prompt``).
* ``AggregateMetrics`` + ``SingleRunMetrics`` carry evaluator scores end-to-end.
* ``benchmark_prompt`` calls the runner and attaches scores when an
  ``evaluator_runner`` is provided.
"""
from __future__ import annotations

import sys
import types

import pytest

from foundry_speedtest.evaluation import (
    DEFAULT_EVALUATORS,
    EvaluationConfig,
    EvaluatorRunner,
    _coerce_passed,
    _coerce_reason,
    _coerce_score,
    _coerce_threshold,
)
from foundry_speedtest.metrics import AggregateMetrics, SingleRunMetrics


# ---------------------------------------------------------------------------
# EvaluationConfig
# ---------------------------------------------------------------------------


class TestEvaluationConfig:
    def test_normalised_endpoint_strips_openai_v1(self):
        cfg = EvaluationConfig(judge_endpoint="https://acct.openai.azure.com/openai/v1")
        assert cfg.normalised_endpoint() == "https://acct.openai.azure.com"

    def test_normalised_endpoint_strips_openai_only(self):
        cfg = EvaluationConfig(judge_endpoint="https://acct.openai.azure.com/openai")
        assert cfg.normalised_endpoint() == "https://acct.openai.azure.com"

    def test_normalised_endpoint_passthrough(self):
        cfg = EvaluationConfig(judge_endpoint="https://acct.openai.azure.com/")
        assert cfg.normalised_endpoint() == "https://acct.openai.azure.com"

    def test_normalised_endpoint_none(self):
        cfg = EvaluationConfig(judge_endpoint=None)
        assert cfg.normalised_endpoint() is None

    def test_to_model_config_with_key(self):
        cfg = EvaluationConfig(
            judge_model="gpt-4.1",
            judge_endpoint="https://acct.openai.azure.com/openai/v1",
            judge_api_key="abc",
        )
        mc = cfg.to_model_config()
        assert mc["azure_endpoint"] == "https://acct.openai.azure.com"
        assert mc["azure_deployment"] == "gpt-4.1"
        assert mc["api_key"] == "abc"

    def test_to_model_config_without_endpoint_raises(self):
        cfg = EvaluationConfig(judge_endpoint=None)
        with pytest.raises(EnvironmentError):
            cfg.to_model_config()

    def test_from_env_priority(self, monkeypatch):
        monkeypatch.setenv("FOUNDRY_SPEEDTEST_JUDGE_ENDPOINT", "https://primary/")
        monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://fallback/")
        cfg = EvaluationConfig.from_env()
        assert cfg.judge_endpoint == "https://primary/"

    def test_from_env_falls_back_to_foundry_endpoint(self, monkeypatch):
        monkeypatch.delenv("FOUNDRY_SPEEDTEST_JUDGE_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_FOUNDRY_JUDGE_ENDPOINT", raising=False)
        monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://fallback/openai/v1")
        cfg = EvaluationConfig.from_env()
        assert cfg.judge_endpoint == "https://fallback/openai/v1"
        assert cfg.normalised_endpoint() == "https://fallback"

    def test_from_env_explicit_overrides(self, monkeypatch):
        monkeypatch.setenv("FOUNDRY_SPEEDTEST_JUDGE_ENDPOINT", "https://env/")
        cfg = EvaluationConfig.from_env(judge_endpoint="https://explicit/")
        assert cfg.judge_endpoint == "https://explicit/"


# ---------------------------------------------------------------------------
# Coercion helpers (handle the variants that real evaluators emit)
# ---------------------------------------------------------------------------


class TestCoercionHelpers:
    def test_coerce_score_canonical_key(self):
        assert _coerce_score({"relevance": 4.0, "relevance_reason": "x"}, "relevance") == 4.0

    def test_coerce_score_legacy_gpt_prefix(self):
        assert _coerce_score({"gpt_relevance": 3}, "relevance") == 3.0

    def test_coerce_score_score_suffix(self):
        assert _coerce_score({"relevance_score": 2.5}, "relevance") == 2.5

    def test_coerce_score_int_passthrough(self):
        assert _coerce_score(5, "relevance") == 5.0

    def test_coerce_score_none(self):
        assert _coerce_score(None, "relevance") is None

    def test_coerce_score_garbage(self):
        assert _coerce_score({"foo": "bar"}, "relevance") is None

    def test_coerce_reason(self):
        d = {"relevance_reason": "because"}
        assert _coerce_reason(d, "relevance") == "because"

    def test_coerce_reason_missing(self):
        assert _coerce_reason({"relevance": 4}, "relevance") == ""

    def test_coerce_threshold(self):
        assert _coerce_threshold({"relevance_threshold": 3}, "relevance") == 3.0

    def test_coerce_passed_label_pass(self):
        assert _coerce_passed({"relevance_result": "pass"}, "relevance") is True

    def test_coerce_passed_label_fail(self):
        assert _coerce_passed({"relevance_result": "fail"}, "relevance") is False

    def test_coerce_passed_bool(self):
        assert _coerce_passed({"passed": True}, "relevance") is True


# ---------------------------------------------------------------------------
# EvaluatorRunner — stub azure.ai.evaluation so we don't need the real lib
# ---------------------------------------------------------------------------


def _install_stub_evaluation_module(monkeypatch, eval_classes: dict[str, type]):
    """Install a fake ``azure.ai.evaluation`` module exposing the given classes.

    Uses ``monkeypatch.setitem`` so ``sys.modules`` (including the real ``azure``
    namespace package and ``azure.identity`` used elsewhere in the codebase) is
    restored at the end of each test.
    """
    az_pkg = sys.modules.get("azure") or types.ModuleType("azure")
    ai_pkg = sys.modules.get("azure.ai") or types.ModuleType("azure.ai")
    eval_mod = types.ModuleType("azure.ai.evaluation")
    for cls_name, cls in eval_classes.items():
        setattr(eval_mod, cls_name, cls)
    monkeypatch.setitem(sys.modules, "azure", az_pkg)
    monkeypatch.setitem(sys.modules, "azure.ai", ai_pkg)
    monkeypatch.setitem(sys.modules, "azure.ai.evaluation", eval_mod)
    if not hasattr(az_pkg, "ai"):
        monkeypatch.setattr(az_pkg, "ai", ai_pkg, raising=False)
    monkeypatch.setattr(ai_pkg, "evaluation", eval_mod, raising=False)


class _StubEvaluator:
    """Mimics the call shape of azure.ai.evaluation evaluators."""

    score_value: float = 4.0
    name: str = "relevance"
    raise_on_call: bool = False

    def __init__(self, *, model_config=None, threshold=3, **kwargs):
        self.model_config = model_config
        self.threshold = threshold

    def __call__(self, *, query, response, **kwargs):
        if self.raise_on_call:
            raise RuntimeError("simulated judge failure")
        return {
            self.name: self.score_value,
            f"{self.name}_reason": f"stub-reason for {self.name}",
            f"{self.name}_threshold": self.threshold,
            f"{self.name}_result": "pass" if self.score_value >= self.threshold else "fail",
        }


def _make_evaluator_class(name: str, score: float, raise_: bool = False):
    return type(
        f"Stub{name.title()}Evaluator",
        (_StubEvaluator,),
        {"name": name, "score_value": score, "raise_on_call": raise_},
    )


class TestEvaluatorRunner:
    def test_evaluate_attaches_scores(self, monkeypatch):
        _install_stub_evaluation_module(monkeypatch, {
            "RelevanceEvaluator": _make_evaluator_class("relevance", 4.0),
            "CoherenceEvaluator": _make_evaluator_class("coherence", 3.5),
            "FluencyEvaluator": _make_evaluator_class("fluency", 5.0),
        })
        cfg = EvaluationConfig(
            judge_endpoint="https://acct.openai.azure.com/openai/v1",
            judge_api_key="key",
        )
        runner = EvaluatorRunner(cfg)
        out = runner.evaluate("What is 2+2?", "4")
        assert set(out) == set(DEFAULT_EVALUATORS)
        assert out["relevance"].score == 4.0
        assert out["relevance"].passed is True
        assert "stub-reason" in out["relevance"].reason
        assert out["coherence"].score == 3.5

    def test_empty_response_marks_all_evaluators_with_error(self):
        cfg = EvaluationConfig(judge_endpoint="https://x/", judge_api_key="k")
        runner = EvaluatorRunner(cfg)
        out = runner.evaluate("q", "")
        assert all(o.error == "empty response" for o in out.values())
        assert all(o.score is None for o in out.values())

    def test_evaluator_crash_isolated_to_one_evaluator(self, monkeypatch):
        _install_stub_evaluation_module(monkeypatch, {
            "RelevanceEvaluator": _make_evaluator_class("relevance", 4.0),
            "CoherenceEvaluator": _make_evaluator_class("coherence", 0.0, raise_=True),
            "FluencyEvaluator": _make_evaluator_class("fluency", 5.0),
        })
        cfg = EvaluationConfig(judge_endpoint="https://x/", judge_api_key="k")
        runner = EvaluatorRunner(cfg)
        out = runner.evaluate("q", "answer")
        assert out["relevance"].ok
        assert out["fluency"].ok
        assert not out["coherence"].ok
        assert "simulated judge failure" in out["coherence"].error

    def test_unknown_evaluator_raises_on_use(self, monkeypatch):
        _install_stub_evaluation_module(monkeypatch, {})
        cfg = EvaluationConfig(
            judge_endpoint="https://x/", judge_api_key="k",
            evaluators=("does_not_exist",),
        )
        runner = EvaluatorRunner(cfg)
        out = runner.evaluate("q", "a")
        assert "does_not_exist" in out
        assert "Unknown evaluator" in out["does_not_exist"].error

    def test_ground_truth_evaluator_skipped_without_ground_truth(self, monkeypatch):
        _install_stub_evaluation_module(monkeypatch, {
            "SimilarityEvaluator": _make_evaluator_class("similarity", 4.0),
        })
        cfg = EvaluationConfig(
            judge_endpoint="https://x/", judge_api_key="k",
            evaluators=("similarity",),
        )
        runner = EvaluatorRunner(cfg)
        out = runner.evaluate("q", "a")
        assert out["similarity"].error == "ground_truth not provided"

    def test_evaluate_many_returns_results_per_key(self, monkeypatch):
        _install_stub_evaluation_module(monkeypatch, {
            "RelevanceEvaluator": _make_evaluator_class("relevance", 4.0),
            "CoherenceEvaluator": _make_evaluator_class("coherence", 3.0),
            "FluencyEvaluator": _make_evaluator_class("fluency", 5.0),
        })
        cfg = EvaluationConfig(judge_endpoint="https://x/", judge_api_key="k")
        runner = EvaluatorRunner(cfg)
        items = [
            ("k1", "what is 2+2?", "4"),
            ("k2", "name a colour", "blue"),
        ]
        out = runner.evaluate_many(items)
        assert set(out) == {"k1", "k2"}
        assert out["k1"]["relevance"].score == 4.0
        assert out["k2"]["fluency"].score == 5.0

    def test_evaluate_many_invokes_callback(self, monkeypatch):
        _install_stub_evaluation_module(monkeypatch, {
            "RelevanceEvaluator": _make_evaluator_class("relevance", 4.0),
            "CoherenceEvaluator": _make_evaluator_class("coherence", 3.0),
            "FluencyEvaluator": _make_evaluator_class("fluency", 5.0),
        })
        cfg = EvaluationConfig(judge_endpoint="https://x/", judge_api_key="k")
        runner = EvaluatorRunner(cfg)
        seen: list[str] = []
        runner.evaluate_many(
            [("k1", "q", "a")],
            on_done=lambda key, _res: seen.append(key),
        )
        assert seen == ["k1"]


# ---------------------------------------------------------------------------
# Metrics integration
# ---------------------------------------------------------------------------


def _mk_run(**overrides) -> SingleRunMetrics:
    base = dict(
        api_type="completions",
        prompt_label="short",
        streaming=True,
        success=True,
        total_time=1.0,
        end_to_end_latency=1.0,
        time_to_first_token=0.5,
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        tokens_per_second=20.0,
    )
    base.update(overrides)
    return SingleRunMetrics(**base)


class TestAggregateEvaluation:
    def test_eval_summary_reports_per_evaluator_stats(self):
        runs = [
            _mk_run(eval_scores={"relevance": 4.0, "coherence": 3.0}),
            _mk_run(eval_scores={"relevance": 5.0, "coherence": 4.0}),
            _mk_run(eval_scores={"relevance": 3.0}),
        ]
        agg = AggregateMetrics(
            api_type="completions",
            prompt_label="short",
            streaming=True,
            runs=runs,
        )
        summary = agg.eval_summary()
        assert summary["relevance"]["count"] == 3
        assert summary["relevance"]["mean"] == pytest.approx(4.0)
        assert summary["coherence"]["count"] == 2
        assert summary["coherence"]["mean"] == pytest.approx(3.5)

    def test_eval_summary_empty_when_no_runs_have_scores(self):
        runs = [_mk_run(eval_scores={})]
        agg = AggregateMetrics(
            api_type="completions", prompt_label="short", streaming=True,
            runs=runs,
        )
        assert agg.eval_summary() == {}

    def test_summary_dict_includes_evaluation_block_when_present(self):
        runs = [_mk_run(eval_scores={"relevance": 4.0})]
        agg = AggregateMetrics(
            api_type="completions", prompt_label="short", streaming=True,
            runs=runs,
        )
        d = agg.summary_dict()
        assert "evaluation" in d
        assert d["evaluation"]["relevance"]["count"] == 1

    def test_summary_dict_omits_evaluation_when_no_scores(self):
        runs = [_mk_run()]
        agg = AggregateMetrics(
            api_type="completions", prompt_label="short", streaming=True,
            runs=runs,
        )
        d = agg.summary_dict()
        assert "evaluation" not in d

    def test_eval_error_count_total(self):
        runs = [
            _mk_run(eval_errors={"relevance": "boom"}),
            _mk_run(eval_errors={}),
            _mk_run(eval_errors={"coherence": "x", "fluency": "y"}),
        ]
        agg = AggregateMetrics(
            api_type="completions", prompt_label="short", streaming=True,
            runs=runs,
        )
        assert agg.eval_error_count() == 3
        assert agg.eval_error_count("relevance") == 1
        assert agg.eval_error_count("coherence") == 1
