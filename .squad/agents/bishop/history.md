# Project Context

- **Owner:** JJ
- **Project:** MS-Foundry-Speedtest
- **Stack:** Python, Azure Identity, OpenAI Python SDK, Rich CLI, Microsoft Foundry deployments.
- **Created:** 2026-05-13T06:57:06Z

## Learnings

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

