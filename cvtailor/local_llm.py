"""Client for a local model behind an OpenAI-compatible server (Ollama, llama.cpp,
LM Studio, vLLM).

It mimics the small part of the Anthropic client that `llm.LLM` uses
(`client.messages.create(...)` returning `.content`, `.stop_reason` and `.usage`), so
prompts, schemas, validation and retries stay the same for every backend. The JSON
schema goes out as `response_format`, which these servers enforce with
grammar-constrained decoding: the model can only emit ids that exist in the master.

Uses only the standard library, so no extra package is needed.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any, Callable, Optional

from .llm import LLMError

_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


class LocalClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 600.0,
                 reasoning: Optional[str] = None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.timeout = timeout
        # Sent as reasoning_effort ("low", "medium", "high"; Ollama also takes "none"
        # for some models). Long thinking is slow on local hardware and can run past
        # the context window.
        self.reasoning = reasoning
        self.accepts_reasoning = True  # LLM may override reasoning per call
        self.accepts_stream = True     # create(on_text=...) streams the answer
        self.messages = self  # so callers can write client.messages.create(...)

    def create(self, *, model: str, max_tokens: int, system: list[dict], messages: list[dict],
               output_config: Optional[dict] = None, reasoning: Optional[str] = None,
               on_text: Optional[Callable[[str], None]] = None, **_ignored: Any) -> SimpleNamespace:
        """on_text, if given, streams the answer: it is called with the text so far
        after every chunk (for a live preview). The result is the same either way."""
        # Anthropic-only fields (cache_control, effort, tools) have no equivalent here
        # and are dropped. The cached system blocks become one system message.
        schema = (output_config or {}).get("format", {}).get("schema")
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": "\n\n".join(b["text"] for b in system)},
                         *messages],
            "stream": False,
        }
        effort = reasoning if reasoning is not None else self.reasoning
        if effort:  # "" means: the model's default
            body["reasoning_effort"] = effort
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": True, "schema": schema},
            }
        data = self._stream(body, on_text) if on_text else self._post(body)
        try:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"local LLM returned an unexpected response: {str(data)[:200]}") from None
        u = data.get("usage") or {}
        return SimpleNamespace(
            # "length" means the answer was cut off; llm._text_of reports it.
            stop_reason="max_tokens" if choice.get("finish_reason") == "length" else "end_turn",
            content=[SimpleNamespace(type="text", text=_THINK.sub("", text))],
            usage=SimpleNamespace(input_tokens=u.get("prompt_tokens", 0),
                                  output_tokens=u.get("completion_tokens", 0)),
        )

    def _stream(self, body: dict, on_text: Callable[[str], None]) -> dict:
        """POST with stream=true (server-sent events) and rebuild the response that a
        non-streamed call returns."""
        body = {**body, "stream": True, "stream_options": {"include_usage": True}}
        text, finish, usage = [], None, {}

        def read(resp: Any) -> None:
            nonlocal finish, usage
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                usage = chunk.get("usage") or usage
                for ch in chunk.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        text.append(piece)
                        on_text("".join(text))
                    finish = ch.get("finish_reason") or finish
        self._request(body, read)
        return {"choices": [{"message": {"content": "".join(text)}, "finish_reason": finish}], "usage": usage}

    def _post(self, body: dict) -> dict:
        out: dict = {}
        self._request(body, lambda resp: out.update(json.loads(resp.read().decode("utf-8"))))
        return out

    def _request(self, body: dict, handle: Callable[[Any], None]) -> None:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                handle(resp)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace").strip()[:300]
            raise LLMError(f"local LLM at {self.url} answered HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            reason = getattr(e, "reason", e)
            raise LLMError(f"could not reach the local LLM at {self.url}: {reason} "
                           "(is the server running, and is CV_LLM_BASE_URL right?)") from None
        except json.JSONDecodeError:
            raise LLMError(f"local LLM at {self.url} did not answer with JSON") from None
