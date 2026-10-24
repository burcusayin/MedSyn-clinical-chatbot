"""Privacy guard for requests that MedSyn sends to OpenRouter.

MedSyn transmits de-identified clinical text from credentialed datasets (MIMIC-IV) to language
models hosted behind the OpenRouter API. PhysioNet's guidance on using credentialed data with
generative-AI services (https://physionet.org/news/post/gpt-responsible-use) does not allow the data
to be sent through third-party APIs that may retain or review it, and names Azure OpenAI (with the
human-review opt-out) as the acceptable route for OpenAI models.

This module builds OpenRouter "provider" routing preferences
(https://openrouter.ai/docs/guides/routing/provider-selection) so that every request is

  * restricted to zero-data-retention endpoints          (``zdr: true``),
  * restricted to providers that do not collect data     (``data_collection: "deny"``), and
  * for OpenAI's proprietary models (openai/gpt-*), pinned to the Azure endpoint with no fallback
    (``order: ["Azure"], allow_fallbacks: false``).

A request that cannot be served under these constraints fails instead of being routed to another
endpoint. That includes models for which OpenRouter lists no zero-data-retention endpoint at all
(for example google/gemini-2.5-flash on 2026-10-24, served only by Google AI Studio, which retains
prompts): such a model cannot be used with clinical text under this guard, by design. The guard is on by default; it can be tuned or disabled with environment variables, which
is NOT recommended when clinical text is involved:

  OPENROUTER_PRIVACY_GUARD    1 (default) | 0  -> 0 disables the guard entirely
  OPENROUTER_ZDR              1 (default) | 0  -> zero-data-retention endpoints only
  OPENROUTER_DATA_COLLECTION  deny (default) | allow
  OPENROUTER_PROVIDER_ORDER   comma-separated provider names to pin; default "Azure" for openai/gpt-*
                              models (not the open-weight gpt-oss-* models) and no pin otherwise;
                              "any" removes the pin
  OPENROUTER_ALLOW_FALLBACKS  0 (default) | 1

History: the guard was added in October 2026, after the sessions reported in the MedSyn paper, which
ran under OpenRouter's default routing. It changes nothing about those sessions; it protects anyone
who runs this code from now on.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

class PrivacyGuardViolation(RuntimeError):
    """OpenRouter served a request outside the pinned providers. Never retry or swallow this."""


DEFAULT_PIN = {"openai": ["Azure"]}   # namespace -> provider order recommended by PhysioNet (applies to openai/gpt-* except the open-weight gpt-oss-* models, which Azure does not serve on OpenRouter)
GUARD_HEADER = {"X-Title": "MedSyn"}


def _truthy(v: Optional[str], default: bool) -> bool:
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def bare_model(model: str) -> str:
    """'openrouter/openrouter/openai/gpt-5.2-chat' -> 'openai/gpt-5.2-chat'."""
    m = model.strip()
    while m.startswith("openrouter/"):
        m = m[len("openrouter/"):]
    return m


def model_namespace(model: str) -> str:
    m = bare_model(model)
    return m.split("/", 1)[0].lower() if "/" in m else ""


def default_pin(model: str) -> list:
    """Provider pin applied when OPENROUTER_PROVIDER_ORDER is unset."""
    m = bare_model(model).lower()
    name = m.split("/", 1)[1] if "/" in m else m
    if model_namespace(model) == "openai" and name.startswith("gpt-") and not name.startswith("gpt-oss"):
        return list(DEFAULT_PIN["openai"])
    return []


def provider_preferences(model: str, env: Optional[Mapping[str, str]] = None) -> Optional[Dict[str, Any]]:
    """OpenRouter 'provider' object for this model, or None when the guard is disabled."""
    env = os.environ if env is None else env
    if not _truthy(env.get("OPENROUTER_PRIVACY_GUARD"), True):
        return None
    prefs: Dict[str, Any] = {}
    if _truthy(env.get("OPENROUTER_ZDR"), True):
        prefs["zdr"] = True
    collection = (env.get("OPENROUTER_DATA_COLLECTION") or "deny").strip().lower()
    if collection not in ("deny", "allow"):
        raise ValueError("OPENROUTER_DATA_COLLECTION must be 'deny' or 'allow'")
    prefs["data_collection"] = collection
    raw_order = env.get("OPENROUTER_PROVIDER_ORDER")
    if raw_order is None or raw_order.strip() == "":
        order = default_pin(model)
    elif raw_order.strip().lower() == "any":
        order = []
    else:
        order = [p.strip() for p in raw_order.split(",") if p.strip()]
    if order:
        prefs["order"] = order
        prefs["allow_fallbacks"] = _truthy(env.get("OPENROUTER_ALLOW_FALLBACKS"), False)
    return prefs


def extra_body(model: str, env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Keyword-ready request-body extension for the openai client: {'provider': {...}} or {}."""
    prefs = provider_preferences(model, env)
    return {"provider": prefs} if prefs else {}


def provider_allowed(provider_name: Optional[str], model: str, env: Optional[Mapping[str, str]] = None) -> bool:
    """True when a response's provider is compatible with the pin (ZDR itself cannot be read from the name)."""
    prefs = provider_preferences(model, env)
    if not prefs or not prefs.get("order"):
        return True
    if not provider_name:
        return False
    return provider_name.strip().lower() in {p.lower() for p in prefs["order"]}


def check_response_provider(provider_name: Optional[str], model: str, env: Optional[Mapping[str, str]] = None) -> None:
    if not provider_allowed(provider_name, model, env):
        raise PrivacyGuardViolation(
            f"OpenRouter served {bare_model(model)} from provider {provider_name!r}, which is outside the pinned "
            f"providers {provider_preferences(model, env).get('order')}. Stopping to avoid sending further data.")


# ── Langroid integration ────────────────────────────────────────────────────────────────────────────

def langroid_call_params(model: str, env: Optional[Mapping[str, str]] = None):
    """OpenAICallParams carrying the routing preferences, or None when the guard is disabled.

    Raises RuntimeError when the installed Langroid cannot carry extra request-body fields, so that
    the application fails closed instead of silently sending unguarded requests.
    """
    body = extra_body(model, env)
    if not body:
        return None
    try:
        from langroid.language_models.openai_gpt import OpenAICallParams
    except Exception as e:  # pragma: no cover
        raise RuntimeError("langroid is required for the OpenRouter privacy guard") from e
    if "extra_body" not in getattr(OpenAICallParams, "model_fields", {}):
        raise RuntimeError(
            "The installed langroid version cannot pass OpenRouter routing preferences (OpenAICallParams has no "
            "'extra_body'); upgrade langroid (>= 0.69.14) or set OPENROUTER_PRIVACY_GUARD=0 (not recommended).")
    return OpenAICallParams(extra_body=body)


def assert_langroid_passthrough(llm, model: str, env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Offline self-check: build the request Langroid would send and verify the 'provider' block survived.

    Returns the provider block. Raises RuntimeError if it was dropped (fail closed). No network call is made.
    """
    expected = provider_preferences(model, env)
    if not expected:
        return {}
    try:
        args = llm._prep_chat_completion(messages="self-check", max_tokens=1)
    except Exception as e:
        raise RuntimeError("Could not verify that Langroid forwards the OpenRouter privacy preferences "
                           f"(langroid internals changed?): {e!r}. Refusing to run unguarded.") from e
    got = (args.get("extra_body") or {}).get("provider")
    if got != expected:
        raise RuntimeError(f"Langroid did not carry the OpenRouter privacy preferences (expected {expected}, request has {got}).")
    return got


# ── Live check (no clinical text is sent) ───────────────────────────────────────────────────────────

def live_check(model: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Send a one-word prompt through the guard and report which provider answered."""
    from openai import OpenAI
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        env_file = Path(__file__).resolve().parents[1] / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set (environment or .env)")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, default_headers=GUARD_HEADER)
    r = client.chat.completions.create(model=bare_model(model), max_tokens=5, temperature=0,
                                       messages=[{"role": "user", "content": "Reply with the single word OK."}],
                                       extra_body=extra_body(model))
    provider = (getattr(r, "model_extra", None) or {}).get("provider")
    check_response_provider(provider, model)
    return {"model": getattr(r, "model", None), "provider": provider, "routing": extra_body(model), "reply": (r.choices[0].message.content or "").strip()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Show or test the OpenRouter privacy routing for a model.")
    ap.add_argument("--model", default="openai/gpt-5.2-chat"); ap.add_argument("--live", action="store_true", help="send a harmless one-word prompt and report the serving provider")
    a = ap.parse_args()
    print("routing preferences:", extra_body(a.model))
    if a.live:
        print(live_check(a.model))
