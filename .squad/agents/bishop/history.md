# Project Context

- **Owner:** JJ
- **Project:** MS-Foundry-Speedtest
- **Stack:** Python, Azure Identity, OpenAI Python SDK, Rich CLI, Microsoft Foundry deployments.
- **Created:** 2026-05-13T06:57:06Z

## Learnings

- Per user directive (2026-05-13), all models on ai-justinjoy-4099 officially support the Responses API. The `supports_responses_api=False` guard was a workaround for a transient service-side failure and has been removed. If HTTP 500s resurface on a specific model, re-introduce a targeted capability override rather than a broad guard.


- The benchmark adapters read `ModelCapabilities.for_model()` to decide which parameters are safe to send per model family.
- `gpt-chat-latest` previously rejected `temperature=0.7` with a 400 and succeeds on Chat Completions when temperature is omitted/defaulted.
- The Responses API path for `gpt-chat-latest` (version `2026-05-05`, resource `ai-justinjoy-4099`) returns HTTP 500 consistently. **Root cause: service-side Foundry backend failure, not a request-shape issue.**

### Investigation: gpt-chat-latest Responses API (2026-05-xx)

**Methodology:** Raw `curl` probes against `https://ai-justinjoy-4099.openai.azure.com/openai/v1/responses` using `az account get-access-token` bearer auth.

**Findings:**
- Chat Completions (`/v1/chat/completions`) succeeds for `gpt-chat-latest` → deployment is provisioned and serving.
- Responses API (`/v1/responses`) returns HTTP 500 for every request shape tried: with/without `instructions`, with/without `store=false`, `max_output_tokens` ranging 16–512.
- Streaming probe revealed the failure signature: `response.created` (status `in_progress`) → `error` event → `response.failed`. The response object is created but inference fails on the backend.
- `gpt-4.1` on the same resource's Responses API works correctly (HTTP 200).
- Microsoft docs list `gpt-chat-latest` version `2026-05-05` as Responses API supported in `swedencentral` — discrepancy is service-side.

**Request IDs captured:**
- `2732981b-ed9b-4947-9709-c66b93a49944`
- `8cf64502-7648-4328-afe5-40bf28890977`
- `3017221e-0420-45de-960a-d2eefe6bb072`
- `89408f48-ed00-4e46-aaad-079c4a41782e`

**Fix applied:** Added `supports_responses_api: bool = True` field to `ModelCapabilities` dataclass; set `False` for the `_uses_default_temperature_only` branch (`gpt-chat-latest`, `gpt-5-chat-latest`). Added early-exit guard in `run_responses()` that returns a descriptive `metrics.error` without making an API call.

**Recommended next step:** File an Azure support ticket referencing the above request IDs, or wait for a service update. When the service is fixed, remove the `supports_responses_api=False` override from `_uses_default_temperature_only` branch in `config.py`.

---

### Investigation: APIM Project Endpoint for Responses API (2026-05-13)

**Context:** JJ provided an APIM Responses endpoint:
`https://apim-yiaefkyinmgwy.azure-api.net/ai-justinjoy-4099/api/projects/ai-justinjoy-4099-project/openai/v1/responses`
Task: determine `base_url` mapping, why the `gpt-chat-latest` guard blocks this path, and the minimal safe integration proposal.

**Finding 1 — `base_url` mapping:**
The OpenAI Python SDK's `client.responses.create()` appends `/responses` to `base_url`. Therefore `base_url` must end at `/openai/v1`:
```
APIM_FOUNDRY_ENDPOINT=https://apim-yiaefkyinmgwy.azure-api.net/ai-justinjoy-4099/api/projects/ai-justinjoy-4099-project/openai/v1
```
The SDK constructs the full URL as `{base_url}/responses`, matching the APIM endpoint exactly. This is the correct form; **do not** pass the full `/responses` path as `base_url`.

**Finding 2 — Authentication mismatch:**
The current `_get_client()` passes an Azure AD bearer token (via `DefaultAzureCredential`) as `api_key`. APIM subscription keys are separate credentials — unless the APIM policy is explicitly configured for AAD passthrough, the Azure AD token will be rejected (401). The APIM key must be supplied via a new env var (`APIM_API_KEY`) as the `api_key` argument to `OpenAI(...)`. This is a secret; never hardcode.

**Finding 3 — Why the guard blocks `gpt-chat-latest` before the APIM endpoint is tried:**
`ModelCapabilities.for_model("gpt-chat-latest")` hits the `_uses_default_temperature_only` branch (exact string match, case-insensitive) and returns `supports_responses_api=False`.
In `run_responses()` (adapters.py L250–258), the guard fires immediately after `ModelCapabilities.for_model()` — before `_get_client()` is even called (L260). The guard is **endpoint-agnostic**: it was coded based on evidence from the direct Foundry endpoint (`ai-justinjoy-4099.openai.azure.com`). The APIM project endpoint routes through `/projects/ai-justinjoy-4099-project/`, a different path that may route to a different inference backend. The guard will silently block any APIM probe for `gpt-chat-latest` without making a call.

**Finding 4 — Minimal safe integration proposal (not yet implemented):**
Three parts, none involving hardcoded secrets:

1. **New env vars** (`.env.example` update is safe/non-secret):
   - `APIM_FOUNDRY_ENDPOINT` — the base URL above (URL only, not a secret)
   - `APIM_API_KEY` — APIM subscription key (secret, env var only)

2. **New client factory** in `adapters.py` (low-risk, purely additive):
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
   ```

3. **Guard bypass in `run_responses()`** (medium-risk, needs team sign-off): Check `APIM_FOUNDRY_ENDPOINT` before applying the `supports_responses_api` gate. If an APIM client is available, skip the guard and use it. The guard was evidence-based on the direct endpoint; the APIM backend may behave differently. Live probe required before committing.

**Decision:** Not implementing the adapters.py change — it requires a live APIM probe to confirm the endpoint works for `gpt-chat-latest` and team consensus on the guard bypass. Filed as decision inbox entry `bishop-apim-responses-endpoint.md`.

---

### Benchmark: gpt-chat-latest vs gpt-4.1 and gpt-5.4-mini (2026-05-13)

**Resource:** ai-justinjoy-4099 (swedencentral) | **Iterations:** 3

**gpt-chat-latest vs gpt-4.1 — gpt-chat-latest wins all metrics:**
- TTFT Mean: 1224ms vs 1435ms
- Total Time Mean: 4753ms vs 4889ms
- TPS Mean: 54.7 vs 50.2

**gpt-chat-latest vs gpt-5.4-mini — gpt-5.4-mini wins all metrics:**
- TTFT Mean: 1344ms vs 1295ms
- Total Time Mean: 4644ms vs 2997ms (gpt-5.4-mini ~35% faster)
- TPS Mean: 55.3 vs 85.3 (gpt-5.4-mini ~54% higher throughput)

**Guard behavior confirmed correct:** gpt-chat-latest Responses API rows all show 100% error rate with 0 tokens and no latency metrics — the `supports_responses_api=False` early-exit is firing cleanly before any HTTP call, producing clean benchmark output. Completions rows show real valid data for gpt-chat-latest throughout.

**Takeaway:** gpt-chat-latest is on par with gpt-4.1 for latency-sensitive completions workloads and is measurably faster. It is outpaced by gpt-5.4-mini on throughput, as expected for a mini-tier model. Decision filed: `.squad/decisions/inbox/bishop-gpt-chat-latest-benchmark.md`.

