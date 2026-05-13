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

