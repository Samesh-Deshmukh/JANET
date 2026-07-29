# src/ai_core/llm.py
"""The one place in JANET that talks to a language model.

Speaks the OpenAI-compatible `/v1/chat/completions` shape over plain HTTP, so the
backend is just a URL:

    llama.cpp  http://127.0.0.1:8081/v1   (default — reuses the GPU-resident server)
    Ollama     http://127.0.0.1:11434/v1

Why not the `ollama` package? Because the owner already runs `llama-server` with
qwen3-14b fully on the GPU. Ollama was loading a *second* copy of the same
weights into the ~4GB that were left, which pushed it 72% onto the CPU: 6-7s per
reply instead of 1.9s, and 46-76s to think instead of 11s. Talking to the loaded
server is both faster and simpler, and this shape works with either backend.
"""
import json
import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_URL = "http://127.0.0.1:8081/v1"
DEFAULT_MODEL = "qwen3-14b"          # llama.cpp serves one model and ignores this

TIMEOUT_S = 45                        # a normal schema-constrained reply (~2-4s)
DEEP_TIMEOUT_S = 180                  # an escalated think-it-through reply (~11s)

# Qwen writes its reasoning inline, before the answer, whenever it ISN'T pinned
# to a JSON schema. That is what "deep thinking" means on this backend.
_THINK = re.compile(r"<think>(.*?)</think>\s*", re.S)


class LLMUnavailable(RuntimeError):
    """Server unreachable, or it returned something we can't use.

    Operational, not a bug — the caller falls back to canned speech.
    """


def _base_url():
    return os.environ.get("JANET_LLM_URL", DEFAULT_URL).rstrip("/")


def _model():
    return os.environ.get("JANET_LLM_MODEL", DEFAULT_MODEL)


def split_thinking(text):
    """Split '<think>reasoning</think>answer' into (thinking, answer).

    Kept here so nothing above this module ever deals in tags.
    """
    match = _THINK.search(text)
    if not match:
        # An OPENING tag with no close means the model ran out of tokens while
        # still reasoning — measured: a code request came back as 17k characters
        # of <think> and no answer at all. Returning that as the "answer" would
        # hand the caller reasoning it would then try to parse (or speak).
        if "<think>" in text:
            return text.strip(), ""
        return "", text.strip()
    return match.group(1).strip(), _THINK.sub("", text, count=1).strip()


def chat(messages, schema=None, max_tokens=400, timeout=TIMEOUT_S):
    """One chat completion.

    With `schema` -> returns the parsed JSON dict (the model is constrained to it,
    and emits no <think> block). Without -> returns (thinking, answer).
    """
    body = {"model": _model(), "messages": messages, "max_tokens": max_tokens}
    if schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "janet_reply", "schema": schema},
        }
    try:
        response = requests.post(
            f"{_base_url()}/chat/completions", json=body, timeout=timeout
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        # ValueError covers a non-JSON body; Key/IndexError an unexpected shape.
        # urllib3's connection errors are a paragraph long — the owner watches
        # this console live, so keep it to one readable line.
        detail = " ".join(str(exc).split())[:90]
        raise LLMUnavailable(f"{type(exc).__name__}: {detail}") from exc

    if schema is None:
        return split_thinking(content)
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMUnavailable(f"model returned non-JSON: {content[:80]!r}") from exc
