"""Tests for the OpenRouter privacy guard (offline). Run: python tests/test_openrouter_privacy.py  (or pytest)."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.openrouter_privacy import provider_preferences, extra_body, provider_allowed, check_response_provider, bare_model, model_namespace

E0 = {}  # no overrides -> defaults


def test_defaults_openai_model_pinned_to_azure_zdr():
    p = provider_preferences("openrouter/openai/gpt-5.2-chat", E0)
    assert p == {"zdr": True, "data_collection": "deny", "order": ["Azure"], "allow_fallbacks": False}
    assert extra_body("openai/gpt-5.2-chat-20251211", E0) == {"provider": p}


def test_double_prefix_and_namespace():
    assert bare_model("openrouter/openrouter/openai/gpt-5.2-chat") == "openai/gpt-5.2-chat"
    assert model_namespace("openrouter/google/gemini-2.5-flash") == "google"
    assert model_namespace("gpt-4o") == ""


def test_non_openai_models_get_zdr_without_pin():
    p = provider_preferences("google/gemini-2.5-flash", E0)
    assert p == {"zdr": True, "data_collection": "deny"}
    assert provider_preferences("openai/gpt-oss-120b", E0) == {"zdr": True, "data_collection": "deny"}   # open-weight: no Azure endpoint, so no pin
    assert provider_preferences("openai/gpt-5.1-chat", E0)["order"] == ["Azure"]
    assert provider_allowed("Google", "google/gemini-2.5-flash", E0)   # no pin -> any provider name accepted


def test_env_overrides():
    e = {"OPENROUTER_PROVIDER_ORDER": "Azure, Google Vertex", "OPENROUTER_ALLOW_FALLBACKS": "1", "OPENROUTER_ZDR": "0", "OPENROUTER_DATA_COLLECTION": "allow"}
    assert provider_preferences("openai/gpt-5.2-chat", e) == {"data_collection": "allow", "order": ["Azure", "Google Vertex"], "allow_fallbacks": True}
    assert provider_preferences("openai/gpt-5.2-chat", {"OPENROUTER_PROVIDER_ORDER": "any"}) == {"zdr": True, "data_collection": "deny"}
    assert provider_preferences("openai/gpt-5.2-chat", {"OPENROUTER_PRIVACY_GUARD": "0"}) is None
    assert extra_body("openai/gpt-5.2-chat", {"OPENROUTER_PRIVACY_GUARD": "false"}) == {}


def test_bad_collection_value_rejected():
    try:
        provider_preferences("openai/gpt-5.2-chat", {"OPENROUTER_DATA_COLLECTION": "maybe"}); assert False, "should raise"
    except ValueError:
        pass


def test_response_provider_check():
    assert provider_allowed("Azure", "openai/gpt-5.2-chat", E0) and provider_allowed("azure", "openai/gpt-5.2-chat", E0)
    assert not provider_allowed("OpenAI", "openai/gpt-5.2-chat", E0) and not provider_allowed(None, "openai/gpt-5.2-chat", E0)
    try:
        check_response_provider("OpenAI", "openai/gpt-5.2-chat", E0); assert False, "should raise"
    except RuntimeError as err:
        assert "OpenAI" in str(err)


def test_langroid_passthrough_if_installed():
    """With Langroid installed, the request Langroid builds must still contain the provider block."""
    try:
        import langroid.language_models as lm
    except Exception:
        print("  (langroid not installed here; pass-through test skipped)"); return
    from src.openrouter_privacy import langroid_call_params, assert_langroid_passthrough
    os.environ.setdefault("OPENROUTER_API_KEY", "sk-or-v1-dummy-for-offline-test")
    for model in ["openrouter/openai/gpt-5.2-chat", "openrouter/openai/gpt-5.2-chat-20251211", "openrouter/google/gemini-2.5-flash", "openrouter/openai/gpt-oss-120b"]:   # gpt-oss: ZDR + deny, no pin
        cfg = lm.OpenAIGPTConfig(chat_model=model, chat_context_length=104_000, seed=42, params=langroid_call_params(model))
        llm = lm.OpenAIGPT(cfg)
        got = assert_langroid_passthrough(llm, model)
        assert got == provider_preferences(model, os.environ), (model, got)
        args = llm._prep_chat_completion(messages="self-check", max_tokens=1)
        assert args["model"] == bare_model(model), args["model"]
        print(f"  langroid {lm.__name__}: {model} -> extra_body.provider = {got}; request model = {args['model']}")
    # fail-closed path: a config without the params must be detected
    cfg = lm.OpenAIGPTConfig(chat_model="openrouter/openai/gpt-5.2-chat", chat_context_length=104_000)
    try:
        assert_langroid_passthrough(lm.OpenAIGPT(cfg), "openrouter/openai/gpt-5.2-chat"); assert False, "should raise"
    except RuntimeError:
        print("  fail-closed check OK (unguarded config is rejected)")


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as e:
                fails += 1; print("FAIL", name, "->", repr(e))
    print("ALL PASSED" if not fails else f"{fails} FAILED"); sys.exit(1 if fails else 0)
