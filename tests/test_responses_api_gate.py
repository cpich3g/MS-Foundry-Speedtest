"""
Unit tests for ModelCapabilities.supports_responses_api gate.

These are deterministic (no live calls). They lock down the logic that
prevents gpt-chat-latest from burning tokens on a guaranteed HTTP 500
from the Foundry Responses API backend.

Evidence from live probe (2026-05-13, resource ai-justinjoy-4099, swedencentral):
  - Responses API returns HTTP 500 for gpt-chat-latest on ALL request shapes:
      bare input only, with instructions, streaming, non-streaming.
  - Chat Completions works normally for gpt-chat-latest.
  - Responses API works for gpt-4.1 (control).
  - Request IDs on file: ba9746e5-..., 3017221e-..., 2ca5641f-...
"""
import pytest

from foundry_speedtest.config import ModelCapabilities, _uses_default_temperature_only


class TestResponsesApiGate:
    """ModelCapabilities.supports_responses_api must be False for chat-latest models."""

    def test_gpt_chat_latest_responses_api_disabled(self):
        caps = ModelCapabilities.for_model("gpt-chat-latest")
        assert caps.supports_responses_api is False, (
            "gpt-chat-latest must have supports_responses_api=False — "
            "service returns HTTP 500 on Foundry (all request shapes, confirmed live)"
        )

    def test_gpt_5_chat_latest_responses_api_not_blocked_by_code(self):
        """gpt-5-chat-latest hits the _is_gpt5 branch first, so responses api is True in code.
        No live evidence it fails on Foundry — do not assert False here."""
        caps = ModelCapabilities.for_model("gpt-5-chat-latest")
        # It matches _is_gpt5 before _uses_default_temperature_only; code intentionally True.
        assert caps.supports_responses_api is True

    def test_gpt_41_responses_api_enabled(self):
        caps = ModelCapabilities.for_model("gpt-4.1")
        assert caps.supports_responses_api is True, (
            "gpt-4.1 Responses API works on Foundry — confirmed live"
        )

    def test_gpt_41_nano_responses_api_enabled(self):
        caps = ModelCapabilities.for_model("gpt-4.1-nano")
        assert caps.supports_responses_api is True

    def test_o4_mini_responses_api_enabled(self):
        caps = ModelCapabilities.for_model("o4-mini")
        # o-series: no temperature, no streaming override, but responses api not blocked
        caps_default = ModelCapabilities.for_model("o4-mini")
        assert caps_default.supports_responses_api is True

    def test_gpt_chat_latest_temperature_disabled(self):
        """Temperature guard must also hold — separate from responses api gate."""
        caps = ModelCapabilities.for_model("gpt-chat-latest")
        assert caps.supports_temperature is False

    def test_uses_default_temperature_only_detection(self):
        assert _uses_default_temperature_only("gpt-chat-latest") is True
        assert _uses_default_temperature_only("GPT-CHAT-LATEST") is True  # case-insensitive
        assert _uses_default_temperature_only("gpt-5-chat-latest") is True
        assert _uses_default_temperature_only("gpt-4.1") is False
        assert _uses_default_temperature_only("gpt-4o") is False


class TestRunResponsesGuard:
    """run_responses must return fail-fast metrics without making an API call for blocked models."""

    def test_run_responses_returns_fail_without_api_call(self, monkeypatch):
        """run_responses for gpt-chat-latest must return failure without touching the client."""
        import foundry_speedtest.adapters as adapters

        called = []

        def fake_get_client():
            called.append(True)
            raise AssertionError("Client must NOT be created for unsupported models")

        monkeypatch.setattr(adapters, "_get_client", fake_get_client)
        # Reset cached client so our patch takes effect
        adapters._client = None

        metrics = adapters.run_responses(
            model="gpt-chat-latest",
            system="You are helpful.",
            user="Hello",
        )

        assert metrics.success is False
        assert not called, "API client was created despite supports_responses_api=False"
        assert "Responses API is not available" in metrics.error or "service-side" in metrics.error.lower() or "HTTP 500" in metrics.error
