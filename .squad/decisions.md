# Squad Decisions

## Active Decisions

### Decision: gpt-chat-latest Responses API is disabled in ModelCapabilities

**Date:** 2026-05-13  
**Authors:** Bishop, Hicks  
**Status:** Implemented — awaiting Microsoft resolution  

#### Summary

`gpt-chat-latest` (version `2026-05-05`) on resource `ai-justinjoy-4099` (swedencentral) has a broken Responses API backend. Deterministic probes confirm:

| API                 | Result        |
|---------------------|---------------|
| Chat Completions    | HTTP 200 ✓    |
| Responses API       | HTTP 500 ✗    |

The failure is service-side (not a request-shape issue). Tried: bare input, with instructions, streaming, non-streaming — all fail. Chat Completions works on same deployment; Responses API works for gpt-4.1 on same resource.

#### Action Taken

1. **Bishop:** Added `supports_responses_api: bool = True` to `ModelCapabilities` dataclass. Set `supports_responses_api=False` in `_uses_default_temperature_only` branch (covers `gpt-chat-latest` and `gpt-5-chat-latest`). Added early-exit guard in `run_responses()` that returns immediately with `success=False` and diagnostic message.

2. **Hicks:** Independently validated with deterministic test suite (`tests/test_responses_api_gate.py`, 8 passed). Fixed code bug: moved `_get_client()` call after the `supports_responses_api` guard to avoid needless client initialization.

#### Request IDs for Azure escalation

- `2732981b-ed9b-4947-9709-c66b93a49944`
- `8cf64502-7648-4328-afe5-40bf28890977`
- `3017221e-0420-45de-960a-d2eefe6bb072`
- `ba9746e5-e925-454d-89ad-042412090cbc`
- `2ca5641f-d470-4754-89a6-a6253e10c6e7`

#### Reverting when fixed

Remove `supports_responses_api=False` from `_uses_default_temperature_only` branch in `foundry_speedtest/config.py` once Microsoft confirms Responses API is functional. Retest with raw curl probe before removing.

## Governance

- All meaningful changes require team consensus
- Document architectural decisions here
- Keep history focused on work, decisions focused on direction
