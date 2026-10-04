import httpx
import pytest

from ragqa.llm import LLMError, OllamaLLM, get_llm


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
