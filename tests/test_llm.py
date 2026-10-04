import json

import httpx
import pytest

from ragqa.llm import GeminiLLM, LLMError, OllamaLLM, get_llm


def client_with(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_complete_sends_the_chat_request_and_returns_the_reply():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"message": {"role": "assistant", "content": " Hello. "}})

    llm = OllamaLLM("tiny", host="localhost:11434", client=client_with(handler))

    assert llm.complete("be brief", "hi") == "Hello."
    assert seen["url"] == "http://localhost:11434/api/chat"
    assert '"model":"tiny"' in seen["body"].replace(" ", "")
    assert '"stream":false' in seen["body"].replace(" ", "")


def test_http_errors_become_llm_errors():
    llm = OllamaLLM(client=client_with(lambda request: httpx.Response(500, text="boom")))
    with pytest.raises(LLMError):
        llm.complete("s", "p")


def test_unreachable_server_becomes_llm_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError, match="Ollama"):
        OllamaLLM(client=client_with(handler)).complete("s", "p")


def test_unexpected_response_shape_becomes_llm_error():
    llm = OllamaLLM(client=client_with(lambda request: httpx.Response(200, json={"oops": 1})))
    with pytest.raises(LLMError):
        llm.complete("s", "p")


@pytest.mark.parametrize(
    "given, expected",
    [
        ("0.0.0.0:11434", "http://localhost:11434"),
        ("http://box:11434/", "http://box:11434"),
        ("box:11434", "http://box:11434"),
    ],
)
def test_host_is_normalized(given, expected):
    assert OllamaLLM(host=given).host == expected


def test_unknown_llm_name_raises():
    with pytest.raises(ValueError):
        get_llm("nope")


GEMINI_REPLY = {
    "id": "x",
    "steps": [
        {"type": "thought", "signature": "abc"},
        {"type": "model_output", "content": [{"type": "text", "text": " Thirty days [1]. "}]},
    ],
}


def test_gemini_sends_system_instruction_and_key_and_reads_the_model_output():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json=GEMINI_REPLY)

    llm = GeminiLLM("tiny", api_key="secret", client=client_with(handler))

    assert llm.complete("be brief", "hi") == "Thirty days [1]."
    assert seen["key"] == "secret"
    assert seen["url"].endswith("/v1beta/interactions")
    assert seen["body"]["model"] == "tiny"
    assert seen["body"]["system_instruction"] == "be brief"
    assert seen["body"]["input"] == "hi"
    assert "secret" not in seen["url"]


def test_gemini_needs_an_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        GeminiLLM()


def test_gemini_reads_the_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert get_llm("gemini").model == "gemini-3.8-flash"


def test_gemini_http_error_includes_status_but_not_the_key():
    llm = GeminiLLM(
        api_key="secret",
        client=client_with(lambda request: httpx.Response(429, text="quota exceeded")),
    )
    with pytest.raises(LLMError) as error:
        llm.complete("s", "p")

    assert "429" in str(error.value)
    assert "quota exceeded" in str(error.value)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "body",
    [{"steps": []}, {"steps": [{"type": "thought"}]}, {"unexpected": True}],
)
def test_gemini_response_without_text_becomes_llm_error(body):
    llm = GeminiLLM(api_key="k", client=client_with(lambda request: httpx.Response(200, json=body)))
    with pytest.raises(LLMError):
        llm.complete("s", "p")


def test_empty_environment_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("RAGQA_LLM", "")
    monkeypatch.setenv("RAGQA_MODEL", "")

    assert get_llm().model == "llama3.2:3b"
