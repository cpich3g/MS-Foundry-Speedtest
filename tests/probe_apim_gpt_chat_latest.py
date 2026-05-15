"""Standalone probe for gpt-chat-latest via APIM gateway.

The `ai-justinjoy-4099` APIM API uses these subscription-key parameter names:
    header: api-key
    query:  subscription-key
The backend forwards to https://ai-justinjoy-4099.services.ai.azure.com/ using
managed identity (resource https://ai.azure.com/), so calling
    /<api-path>/api/projects/<project>/openai/v1/<op>
on APIM is equivalent to calling the project endpoint directly.

Usage:
    python tests/probe_apim_gpt_chat_latest.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

APIM_BASE = os.getenv(
    "AZURE_FOUNDRY_GATEWAY_ENDPOINT",
    "https://apim-yiaefkyinmgwy.azure-api.net/ai-justinjoy-4099/api/projects/ai-justinjoy-4099-project/openai/v1",
).rstrip("/")
APIM_KEY = (
    os.getenv("AZURE_FOUNDRY_GATEWAY_SUBSCRIPTION_KEY")
    or os.getenv("AZURE_FOUNDRY_APIM_SUBSCRIPTION_KEY")
    or os.getenv("AZURE_FOUNDRY_GATEWAY_KEY")
    or os.getenv("APIM_SUBSCRIPTION_KEY")
)
MODEL = os.getenv("PROBE_MODEL", "gpt-chat-latest")

if not APIM_KEY:
    print("ERROR: no APIM subscription key found in env")
    sys.exit(2)

print(f"BASE:  {APIM_BASE}")
print(f"MODEL: {MODEL}")
print(f"KEY:   {APIM_KEY[:6]}…{APIM_KEY[-4:]}")
print()

PAYLOAD_RESPONSES_STRING: dict[str, Any] = {
    "model": MODEL,
    "input": "Say 'hello' in one word.",
    "max_output_tokens": 32,
}
PAYLOAD_RESPONSES_MINIMAL: dict[str, Any] = {
    "model": MODEL,
    "input": "Say 'hello' in one word.",
}
PAYLOAD_RESPONSES_MESSAGES: dict[str, Any] = {
    "model": MODEL,
    "input": [
        {"role": "user", "content": [{"type": "input_text", "text": "Say 'hello' in one word."}]}
    ],
    "max_output_tokens": 32,
}
PAYLOAD_RESPONSES_WITH_INSTRUCTIONS: dict[str, Any] = {
    "model": MODEL,
    "instructions": "You are concise.",
    "input": "Say 'hello' in one word.",
    "max_output_tokens": 32,
}
PAYLOAD_RESPONSES_STORE_FALSE: dict[str, Any] = {
    "model": MODEL,
    "input": "Say 'hello' in one word.",
    "max_output_tokens": 32,
    "store": False,
}
PAYLOAD_RESPONSES_STREAMING: dict[str, Any] = {
    "model": MODEL,
    "input": "Say 'hello' in one word.",
    "max_output_tokens": 32,
    "stream": True,
}
PAYLOAD_RESPONSES_TEMPERATURE: dict[str, Any] = {
    "model": MODEL,
    "input": "Say 'hello' in one word.",
    "max_output_tokens": 32,
    "temperature": 0.7,
}
PAYLOAD_CHAT: dict[str, Any] = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Say 'hello' in one word."}],
    "max_completion_tokens": 32,
}


def _short(text: str, n: int = 260) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def call(label: str, *, url: str, headers: dict[str, str], params: dict[str, str] | None, payload: dict[str, Any]) -> None:
    print(f"--- {label}")
    print(f"    POST {url}")
    masked = {k: ("***" if k.lower() in ("authorization", "ocp-apim-subscription-key", "api-key") else v) for k, v in headers.items()}
    print(f"    headers: {masked}")
    if params:
        masked_q = {k: ("***" if "key" in k.lower() else v) for k, v in params.items()}
        print(f"    query:   {masked_q}")
    print(f"    payload: {json.dumps({k: v for k, v in payload.items() if k != 'input'})}, input=<...>")
    start = time.perf_counter()
    try:
        is_stream = bool(payload.get("stream"))
        with httpx.stream("POST", url, headers=headers, params=params, json=payload, timeout=60.0) as r:
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(f"    status:  {r.status_code}  ({elapsed_ms:.0f} ms)")
            if r.status_code != 200:
                body = r.read().decode("utf-8", errors="replace")
                print(f"    body:    {_short(body, 360)}")
            elif is_stream:
                events = []
                for line in r.iter_lines():
                    if line.startswith("data:"):
                        events.append(line[: 80])
                    if len(events) >= 5:
                        break
                print(f"    stream events (first 5):")
                for e in events:
                    print(f"      {e}")
            else:
                body = r.read().decode("utf-8", errors="replace")
                try:
                    data = json.loads(body)
                    output = (
                        data.get("output_text")
                        or data.get("output")
                        or data.get("choices")
                    )
                    print(f"    output:  {_short(json.dumps(output))}")
                except Exception:
                    print(f"    body:    {_short(body)}")
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"    EXC:     {exc!r}  ({elapsed_ms:.0f} ms)")
    print()


def main() -> None:
    base_headers = {"Content-Type": "application/json"}
    url_responses = f"{APIM_BASE}/responses"
    url_chat = f"{APIM_BASE}/chat/completions"

    print("=" * 60)
    print("Sanity: Chat Completions via APIM (proves gateway works)")
    print("=" * 60)
    call(
        "C1. chat/completions, api-key header",
        url=url_chat,
        headers={**base_headers, "api-key": APIM_KEY},
        params=None,
        payload=PAYLOAD_CHAT,
    )
    call(
        "C2. chat/completions, subscription-key query",
        url=url_chat,
        headers=base_headers,
        params={"subscription-key": APIM_KEY},
        payload=PAYLOAD_CHAT,
    )

    print("=" * 60)
    print("Responses via APIM, payload = string input")
    print("=" * 60)
    call(
        "R1. responses, api-key header",
        url=url_responses,
        headers={**base_headers, "api-key": APIM_KEY},
        params=None,
        payload=PAYLOAD_RESPONSES_STRING,
    )
    call(
        "R2. responses, subscription-key query",
        url=url_responses,
        headers=base_headers,
        params={"subscription-key": APIM_KEY},
        payload=PAYLOAD_RESPONSES_STRING,
    )

    print("=" * 60)
    print("Responses via APIM, payload = messages input")
    print("=" * 60)
    call(
        "R3. responses, api-key header (messages)",
        url=url_responses,
        headers={**base_headers, "api-key": APIM_KEY},
        params=None,
        payload=PAYLOAD_RESPONSES_MESSAGES,
    )

    print("=" * 60)
    print("Responses via APIM, payload = minimal (no max_output_tokens)")
    print("=" * 60)
    call(
        "R4. responses, api-key header, minimal payload",
        url=url_responses,
        headers={**base_headers, "api-key": APIM_KEY},
        params=None,
        payload=PAYLOAD_RESPONSES_MINIMAL,
    )

    print("=" * 60)
    print("Responses via APIM, parameter coverage")
    print("=" * 60)
    call(
        "R5. responses + instructions",
        url=url_responses,
        headers={**base_headers, "api-key": APIM_KEY},
        params=None,
        payload=PAYLOAD_RESPONSES_WITH_INSTRUCTIONS,
    )
    call(
        "R6. responses + store=false",
        url=url_responses,
        headers={**base_headers, "api-key": APIM_KEY},
        params=None,
        payload=PAYLOAD_RESPONSES_STORE_FALSE,
    )
    call(
        "R7. responses + temperature=0.7",
        url=url_responses,
        headers={**base_headers, "api-key": APIM_KEY},
        params=None,
        payload=PAYLOAD_RESPONSES_TEMPERATURE,
    )
    call(
        "R8. responses streaming",
        url=url_responses,
        headers={**base_headers, "api-key": APIM_KEY, "Accept": "text/event-stream"},
        params=None,
        payload=PAYLOAD_RESPONSES_STREAMING,
    )

    print("=" * 60)
    print("Responses via APIM, alternate model names")
    print("=" * 60)
    for alt in ("chatgpt-4o-latest", "gpt-4o", "gpt-4.1", "gpt-5.4"):
        alt_payload = {**PAYLOAD_RESPONSES_STRING, "model": alt}
        call(
            f"R9.{alt}. responses with model='{alt}'",
            url=url_responses,
            headers={**base_headers, "api-key": APIM_KEY},
            params=None,
            payload=alt_payload,
        )


if __name__ == "__main__":
    main()

