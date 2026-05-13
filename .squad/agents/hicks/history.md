# Project Context

- **Owner:** JJ
- **Project:** MS-Foundry-Speedtest
- **Stack:** Python, Azure Identity, OpenAI Python SDK, Rich CLI, Microsoft Foundry deployments.
- **Created:** 2026-05-13T06:57:06Z

## Learnings

- The compare table previously showed a false winner when one model had no successful metric data because `_global_avg()` returned `0`.
- Live validation is appropriate for Foundry deployment-specific behavior, but code regressions should be covered with local tests where possible.
- **gpt-chat-latest Responses API is a service-side gap (confirmed 2026-05-13).** Every request shape fails with HTTP 500 (bare input, with instructions, streaming, non-streaming). Chat Completions works. Responses API works for gpt-4.1. Request IDs: ba9746e5-e925-454d-89ad-042412090cbc, 3017221e-0420-45de-960a-d2eefe6bb072, 2ca5641f-d470-4754-89a6-a6253e10c6e7. Escalate via Azure support ticket.
- **`_get_client()` must be called after capability guards** — calling it before a fail-fast guard wastes connection setup for guaranteed failures. Fixed in `run_responses()` on 2026-05-13.
- **gpt-5-chat-latest hits `_is_gpt5` before `_uses_default_temperature_only`** in `ModelCapabilities.for_model`. Not tested against Responses API live; do not assert it as broken without a probe.
- **Blind benchmark JJ task (gpt-5.5 vs opus-4.7) — deployment name failures.** Neither `gpt-5.5` nor `opus-4.7` exists on `ai-justinjoy-4099`. Error: `DeploymentNotFound` (HTTP 404). Substituted `gpt-5.4` (closest gpt-5.x in version sequence) and `gpt-5.4-mini`. No Anthropic/Claude deployments on this Foundry resource — no `opus-4.7` equivalent; `gpt-5.4-pro` exists but returns HTTP 400 "unsupported operation" for chat completions.
- **AZURE_FOUNDRY_ENDPOINT must include `/openai/v1` path.** When set to just `https://ai-justinjoy-4099.cognitiveservices.azure.com/`, the `OpenAI(base_url=...)` SDK constructs wrong URLs (`/chat/completions` instead of `/openai/v1/chat/completions`), causing every call to fail with 100% error rate. Correct value: `https://ai-justinjoy-4099.cognitiveservices.azure.com/openai/v1`.
- **gpt-5.4 vs gpt-5.4-mini blind benchmark results (3 iters, short/medium/long, completions+responses, streaming+non-streaming):**
  - TTFT Mean: gpt-5.4=1417ms, gpt-5.4-mini=883ms → **mini wins**
  - Total Time Mean: gpt-5.4=5857ms, gpt-5.4-mini=2709ms → **mini wins**
  - TPS Mean: gpt-5.4=41.9, gpt-5.4-mini=94.0 → **mini wins**
  - All three metrics: gpt-5.4-mini is faster. gpt-5.4 produces longer output per response (avg 512 out vs ~11 for short prompts) suggesting higher quality/verbosity, not captured in these speed metrics.
- **supports_responses_api guard removed for gpt-chat-latest (2026-05-13).** User confirmed all models now support Responses API. Updated tests: `test_gpt_chat_latest_responses_api_disabled` → `test_gpt_chat_latest_responses_api_enabled` (asserts True). Deleted `TestRunResponsesGuard`; replaced with `TestRunResponsesNoGuard` which verifies the client IS called. 8 passed, 0 failed.

