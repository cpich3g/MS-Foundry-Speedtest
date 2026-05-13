# Work Routing

How to decide who handles what.

## Routing Table

| Work Type | Route To | Examples |
|-----------|----------|----------|
| Scope, architecture, reviews | Ripley | Trade-offs, reviewer gates, cross-cutting CLI/API decisions |
| Foundry/OpenAI API implementation | Bishop | Adapter behavior, auth, deployment compatibility, CLI plumbing |
| Testing and verification | Hicks | Repro scripts, regression coverage, live validation, edge cases |
| Documentation and user-facing notes | Burke | README updates, usage guidance, concise findings |
| Session logging | Scribe | Automatic session logs and decision merging |
| Work monitoring | Ralph | Backlog checks, open issue monitoring, keep-alive |

## Issue Routing

| Label | Action | Who |
|-------|--------|-----|
| `squad` | Triage: analyze issue, assign `squad:{member}` label | Ripley |
| `squad:ripley` | Pick up architecture/review issue | Ripley |
| `squad:bishop` | Pick up backend/API issue | Bishop |
| `squad:hicks` | Pick up test/QA issue | Hicks |
| `squad:burke` | Pick up docs issue | Burke |

## Rules

1. Route Foundry/OpenAI adapter failures to Bishop first, with Hicks validating live repros.
2. Route false metrics, test failures, and regression coverage to Hicks.
3. Route architectural trade-offs or reviewer rejection gates to Ripley.
4. Route docs and user-facing explanation work to Burke.
5. Scribe runs after substantial work to merge decisions and write logs.

