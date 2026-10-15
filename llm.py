"""LLM client used by every agent.

Gemini and Groq both expose an OpenAI-compatible endpoint, so one client covers both.
Agents ask for JSON; this module extracts and validates it and retries once with the
parse error if the model returns something malformed.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-2.5-flash", "gemini-3.8-flash", "gemini-3.5-flash"],
        "key_name": "GEMINI_API_KEY",
        "label": "Google Gemini",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-8b-instant"],
        "key_name": "GROQ_API_KEY",
        "label": "Groq (Llama 3.3 70B)",
    },
}


class LLMError(RuntimeError):
    pass


def extract_json(text: str):
    """Pull the first JSON object/array out of a model reply (handles ```json fences)."""
    if text is None:
        raise ValueError("empty reply")
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (t.find("{"), t.find("[")) if i != -1]
    if not starts:
        raise ValueError("no JSON object found in reply")
    start = min(starts)
    end = max(t.rfind("}"), t.rfind("]"))
    return json.loads(t[start : end + 1])


@dataclass
class LLM:
    provider: str
    api_key: str
    model: str | None = None
    calls: int = 0
    _lock: object = field(default_factory=__import__("threading").Lock, repr=False)
    total_tokens: int = 0
    _client: object = field(default=None, repr=False)

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise LLMError(f"Unknown provider '{self.provider}'. Use one of: {', '.join(PROVIDERS)}")
        cfg = PROVIDERS[self.provider]
        self._fallbacks = [m for m in cfg["models"] if m != self.model]
        self.model = self.model or self._fallbacks.pop(0)
        from openai import OpenAI

        self._client = OpenAI(api_key=self.api_key, base_url=cfg["base_url"], timeout=40, max_retries=1)

    @property
    def label(self) -> str:
        return f"{PROVIDERS[self.provider]['label']} · {self.model}"

    def chat(self, system: str, user: str, temperature: float = 0.3) -> str:
        while True:
            model = self.model
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=temperature,
                )
                break
            except Exception as e:
                msg = str(e).lower()
                retired = ("404" in msg or "not found" in msg or "does not exist" in msg
                           or "decommissioned" in msg or "model_not_found" in msg)
                # The OpenAI SDK has already retried 5xx/429 twice with backoff at this point.
                busy = any(k in msg for k in ("503", "502", "500", "504", "unavailable", "overloaded",
                                              "high demand", "429", "resource_exhausted",
                                              "timed out", "timeout"))
                with self._lock:  # agents may run in parallel threads
                    if self.model != model and (retired or busy):
                        continue  # another thread already switched models
                    if (retired or busy) and self._fallbacks:
                        self.model = self._fallbacks.pop(0)  # retired or overloaded model: try the next one
                        continue
                self._raise(e)
        with self._lock:
            self.calls += 1
        usage = getattr(resp, "usage", None)
        if usage is not None and getattr(usage, "total_tokens", None):
            self.total_tokens += usage.total_tokens
        return resp.choices[0].message.content or ""

    @staticmethod
    def _raise(e):
        msg = str(e)
        if "401" in msg or "API key" in msg or "PERMISSION_DENIED" in msg:
            raise LLMError("The API key was rejected. Check that it is correct and active.") from e
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate" in msg.lower():
            raise LLMError("The LLM rate limit was hit. Wait a minute and try again.") from e
        if "503" in msg or "UNAVAILABLE" in msg or "high demand" in msg:
            raise LLMError("The AI model is overloaded right now (Google-side, temporary). "
                           "Wait a minute and press Run again.") from e
        raise LLMError(f"LLM request failed: {msg[:300]}") from e

    def json(self, system: str, user: str, temperature: float = 0.3, required: tuple = ()):
        """Ask for JSON, validate required top-level keys, retry once with the error."""
        sys_prompt = system + "\n\nReply with ONE valid JSON object only. No markdown, no commentary."
        reply = self.chat(sys_prompt, user, temperature)
        for attempt in range(2):
            try:
                data = extract_json(reply)
                if not isinstance(data, dict):
                    raise ValueError("top-level JSON must be an object")
                missing = [k for k in required if k not in data]
                if missing:
                    raise ValueError(f"missing keys: {missing}")
                return data
            except (ValueError, json.JSONDecodeError) as e:
                if attempt == 1:
                    raise LLMError(f"The model returned invalid JSON twice ({e}).") from e
                reply = self.chat(
                    sys_prompt,
                    user + f"\n\nYour previous reply could not be used because: {e}. "
                    "Return the complete JSON object again, valid this time.",
                    0.1,
                )
        raise LLMError("unreachable")


def _read_setting(name: str, secrets=None):
    if secrets is not None:
        try:
            if name in secrets:
                return str(secrets[name])
        except Exception:
            pass
    return os.environ.get(name)


def build_llm(secrets=None, override_provider=None, override_key=None):
    """Create the LLM from Streamlit secrets / env vars. Returns None if no key is configured."""
    if _read_setting("CAREERPILOT_MOCK", secrets) == "1":
        from tests.mock_llm import MockLLM

        return MockLLM()
    provider = override_provider or _read_setting("LLM_PROVIDER", secrets) or "gemini"
    key = override_key or _read_setting(PROVIDERS.get(provider, {}).get("key_name", ""), secrets)
    if not key:
        # fall back to whichever provider has a key
        for p, cfg in PROVIDERS.items():
            k = _read_setting(cfg["key_name"], secrets)
            if k:
                provider, key = p, k
                break
    if not key:
        return None
    model = _read_setting("LLM_MODEL", secrets)
    return LLM(provider=provider, api_key=key, model=model)
