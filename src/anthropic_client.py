"""Shared Anthropic Messages API client. Zero third-party imports.

Direct HTTPS via urllib.request -- no SDK. Used by every pipeline stage
that makes an LLM call.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ENV_PATH = Path(".env")
API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
MODEL = "claude-sonnet-4-6"
PLACEHOLDER_KEY = "sk-ant-replace-me"

TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504, 529}  # rate limit, server error, overloaded


def load_env(path: Path = ENV_PATH) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def check_api_key() -> None:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key or key == PLACEHOLDER_KEY:
        sys.exit("ANTHROPIC_API_KEY is missing or still the placeholder in .env -- set a real key before running.")


load_env()
check_api_key()


def call_anthropic(system_prompt: str, user_prompt: str, max_tokens: int) -> tuple[str, str]:
    payload = json.dumps(
        {
            "model": MODEL,
            "max_tokens": max_tokens,
            "temperature": 0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
    ).encode("utf-8")

    attempts = 0
    while True:
        req = urllib.request.Request(
            API_URL,
            data=payload,
            method="POST",
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if attempts >= 1 or e.code not in TRANSIENT_STATUS_CODES:
                raise RuntimeError(f"Anthropic API error {e.code}: {body}") from e
        except (urllib.error.URLError, TimeoutError):
            if attempts >= 1:
                raise
        attempts += 1
        time.sleep(2)

    response_text = "".join(b["text"] for b in data["content"] if b.get("type") == "text")
    return response_text, data["stop_reason"]
