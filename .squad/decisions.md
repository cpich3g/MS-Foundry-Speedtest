# Squad Decisions

## Active Decisions

No decisions recorded yet.
### Decision: gpt-chat-latest Responses API is disabled in ModelCapabilities

**Date:** 2026-05-13  
**Authors:** Bishop, Hicks  
**Status:** Implemented — awaiting Microsoft resolution

#### Benchmark Notes (2026-05-13)

Bishop benchmarked `gpt-chat-latest` against `gpt-4.1` and `gpt-5.4-mini` with 3 iterations per model per prompt size on resource `ai-justinjoy-4099` (swedencentral). Key findings:

- **vs gpt-4.1:** gpt-chat-latest wins all three metrics (TTFT: 1224ms vs 1435ms, Total Time: 4753ms vs 4889ms, TPS: 54.7 vs 50.2).
- **vs gpt-5.4-mini:** gpt-5.4-mini wins all three metrics (expected for mini-tier: 35% faster total time, 54% higher TPS).
- **Guard correctness:** All 6 gpt-chat-latest Responses API rows (across both compares) show 100% error rate, 0 tokens, no latency metrics — correct guarded behavior. No HTTP 500s surfaced; guard is working cleanly.
- **Takeaway:** gpt-chat-latest is on par with gpt-4.1 for completions latency and faster; outclassed by gpt-5.4-mini on throughput.

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

### Decision: Deployment Name Failures + Endpoint Configuration (2026-05-13)

**Date:** 2026-05-13  
**Author:** Hicks  
**Requested by:** JJ  
**Status:** Findings for team review

#### Findings

1. **Requested deployments do not exist:**
   - `gpt-5.5` → 404 DeploymentNotFound
   - `opus-4.7` → 404 DeploymentNotFound (no Anthropic/Claude deployed on resource)

2. **Suggested mappings:**
   - `gpt-5.5` → use `gpt-5.4` (highest gpt-5.x with working chat completions)
   - `opus-4.7` → no equivalent (consider `grok-4` or `o3-pro` if non-OpenAI flagship intended)

3. **Endpoint configuration finding (critical):**
   - `AZURE_FOUNDRY_ENDPOINT` without `/openai/v1` suffix causes 100% error rate (wrong URL construction).
   - Correct form: `https://ai-justinjoy-4099.cognitiveservices.azure.com/openai/v1`
   - **Recommendation:** Update `.env.example` and docs to show `/openai/v1` suffix. Consider adding endpoint validation check in `_get_client()`.

4. **Substitute benchmark results (gpt-5.4 vs gpt-5.4-mini, 3 iterations, 0% error):**
   - TTFT Mean: 1417ms vs 883ms → gpt-5.4-mini wins
   - Total Time Mean: 5857ms vs 2709ms → gpt-5.4-mini wins (~2× faster)
   - TPS Mean: 41.9 vs 94.0 → gpt-5.4-mini wins (~2.2× higher throughput)

#### Actions for JJ

1. Confirm intended model for `opus-4.7` (no Anthropic option available).
2. Update `AZURE_FOUNDRY_ENDPOINT` to include `/openai/v1` suffix.
3. Re-run `gpt-5.5` comparison when deployment becomes available.

**Date:** 2026-05-13  
**Author:** Bishop  
**Status:** Proposed — awaiting live probe and team sign-off  
**Requested by:** JJ

#### Context

JJ provided an APIM Responses endpoint with two questions: (1) how to wire this into the OpenAI SDK `base_url`, and (2) whether the existing `gpt-chat-latest` Responses API guard correctly applies to this new endpoint.

#### Findings

**1. `base_url` — use `/openai/v1`, not the full path**

The OpenAI Python SDK constructs endpoint URL as `{base_url}/{operation}`.
- `client.responses.create()` → `{base_url}/responses`
- `client.chat.completions.create()` → `{base_url}/chat/completions`

**Correct `base_url`:**
```
https://apim-yiaefkyinmgwy.azure-api.net/ai-justinjoy-4099/api/projects/ai-justinjoy-4099-project/openai/v1
```

**2. Authentication: Azure AD token ≠ APIM subscription key**

Current `_get_client()` passes Azure AD bearer token (from `DefaultAzureCredential`) as `api_key`. APIM subscription keys are separate. Unless explicitly configured for Azure AD passthrough, this returns HTTP 401. APIM key must be:
- Stored as env var `APIM_API_KEY` (never hardcoded)
- Passed as `api_key=` when constructing APIM `OpenAI(...)` client

**3. The `gpt-chat-latest` guard blocks before APIM is ever tried**

In `run_responses()` (adapters.py L250–258), the guard fires before `_get_client()` (L260). Guard is endpoint-agnostic — empirically based on direct Foundry endpoint (`ai-justinjoy-4099.openai.azure.com`). APIM project endpoint uses path `/projects/ai-justinjoy-4099-project/`, routing through different backend. Guard incorrectly prevents APIM probe from running.

#### Proposed Code Changes

**Non-secret, low-risk: update `.env.example`**
```dotenv
# Optional: APIM project endpoint (URL only — not a secret)
APIM_FOUNDRY_ENDPOINT=https://apim-yiaefkyinmgwy.azure-api.net/ai-justinjoy-4099/api/projects/ai-justinjoy-4099-project/openai/v1

# Optional: APIM subscription key (SECRET — env var only, never commit)
APIM_API_KEY=
```

**Additive, low-risk: new APIM client factory in `adapters.py`**
```python
_apim_client: OpenAI | None = None

def _get_apim_client() -> OpenAI | None:
    global _apim_client
    if _apim_client is None:
        endpoint = os.getenv("APIM_FOUNDRY_ENDPOINT")
        api_key = os.getenv("APIM_API_KEY")
        if not endpoint or not api_key:
            return None
        _apim_client = OpenAI(base_url=endpoint, api_key=api_key)
    return _apim_client

def reset_apim_client() -> None:
    global _apim_client
    _apim_client = None
```

**Medium-risk: guard bypass in `run_responses()` — needs live probe first**
```python
def run_responses(model, system, user, ...) -> SingleRunMetrics:
    caps = ModelCapabilities.for_model(model)
    metrics = SingleRunMetrics(api_type="responses", ...)

    apim_client = _get_apim_client()

    # Only apply guard when NOT using APIM endpoint
    if not caps.supports_responses_api and apim_client is None:
        metrics.success = False
        metrics.error = "Responses API not available for this model..."
        return metrics

    client = apim_client if apim_client is not None else _get_client()
    ...
```

#### Prerequisites Before Implementing Guard Bypass

1. **Live probe** against APIM endpoint with minimal `gpt-chat-latest` Responses API call — must confirm HTTP 200, not 500.
2. **Auth verification** — confirm APIM expects subscription key or Azure AD passthrough.
3. **Team sign-off** on guard bypass logic.

#### Reverting

If APIM endpoint also returns HTTP 500 for `gpt-chat-latest`, guard bypass is unnecessary. Keep behavior unchanged and log as confirmed dead path.

## Governance

- All meaningful changes require team consensus
- Document architectural decisions here
- Keep history focused on work, decisions focused on direction
