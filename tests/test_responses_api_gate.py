"""
Unit tests for ModelCapabilities.supports_responses_api flag.

These are deterministic (no live calls). They lock down capability detection
logic so parameter-selection regressions are caught before hitting the API.

Note (2026-05-13): gpt-chat-latest previously had supports_responses_api=False
due to HTTP 500 failures on Foundry. That guard was removed after the user
confirmed all models now support the Responses API.
"""
import pytest

from foundry_speedtest.config import ModelCapabilities, _uses_default_temperature_only


class TestResponsesApiGate:
    """ModelCapabilities.supports_responses_api must be True for all supported models."""

    def test_gpt_chat_latest_responses_api_enabled(self):
        caps = ModelCapabilities.for_model("gpt-chat-latest")
        assert caps.supports_responses_api is True, (
            "gpt-chat-latest must have supports_responses_api=True — "
            "all models confirmed to support Responses API"
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


class TestRunResponsesNoGuard:
    """run_responses must attempt an API call for all models (no early-exit guard)."""

    def test_run_responses_attempts_api_call_for_gpt_chat_latest(self, monkeypatch):
        """run_responses for gpt-chat-latest must call the client — guard was removed."""
        import foundry_speedtest.adapters as adapters

        called = []

        class FakeResponses:
            def create(self, **kwargs):
                called.append(kwargs)
                raise RuntimeError("fake-api-error")

        class FakeClient:
            responses = FakeResponses()

        def fake_get_responses_client():
            return FakeClient()

        monkeypatch.setattr(adapters, "_get_responses_client", fake_get_responses_client)
        adapters._client = None
        adapters._responses_client = None

        metrics = adapters.run_responses(
            model="gpt-chat-latest",
            system="You are helpful.",
            user="Hello",
        )

        assert called, "run_responses must call client.responses.create for gpt-chat-latest"
        assert metrics.success is False  # fake error propagates
        assert "fake-api-error" in metrics.error


class TestResponsesEndpointSelection:
    """Responses can use a project or gateway endpoint without breaking Completions."""

    def test_project_endpoint_is_normalized_for_responses(self, monkeypatch):
        import foundry_speedtest.adapters as adapters

        monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://example.openai.azure.com/openai/v1")
        monkeypatch.setenv(
            "AZURE_FOUNDRY_PROJECT_ENDPOINT",
            "https://example.services.ai.azure.com/api/projects/demo-project",
        )
        monkeypatch.delenv("AZURE_FOUNDRY_RESPONSES_TOKEN_SCOPE", raising=False)
        monkeypatch.delenv("AZURE_FOUNDRY_TOKEN_SCOPE", raising=False)

        settings = adapters._client_settings(responses=True)

        assert settings.base_url == (
            "https://example.services.ai.azure.com/api/projects/demo-project/openai/v1"
        )
        assert settings.token_scope == "https://ai.azure.com/.default"

    def test_direct_endpoint_still_used_for_completions(self, monkeypatch):
        import foundry_speedtest.adapters as adapters

        monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://example.openai.azure.com/openai/v1")
        monkeypatch.setenv(
            "AZURE_FOUNDRY_PROJECT_ENDPOINT",
            "https://example.services.ai.azure.com/api/projects/demo-project",
        )
        monkeypatch.delenv("AZURE_FOUNDRY_TOKEN_SCOPE", raising=False)

        settings = adapters._client_settings(responses=False)

        assert settings.base_url == "https://example.openai.azure.com/openai/v1"
        assert settings.token_scope == "https://cognitiveservices.azure.com/.default"

    def test_apim_gateway_key_defaults_to_query_parameter(self, monkeypatch):
        import foundry_speedtest.adapters as adapters

        monkeypatch.setenv(
            "AZURE_FOUNDRY_RESPONSES_ENDPOINT",
            "https://gateway.azure-api.net/resource/api/projects/project/openai/v1",
        )
        monkeypatch.setenv("APIM_SUBSCRIPTION_KEY", "secret-key")
        monkeypatch.delenv("AZURE_FOUNDRY_GATEWAY_KEY_LOCATION", raising=False)

        settings = adapters._client_settings(responses=True)

        assert settings.base_url == "https://gateway.azure-api.net/resource/api/projects/project/openai/v1"
        assert settings.default_query == {"subscription-key": "secret-key"}
        assert settings.default_headers == {}
