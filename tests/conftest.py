import json

import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from app import main
from app.models import Intent


def choice_payload(intent: Intent, confidence: float = 0.95) -> dict:
    """Test double: a System One response, not a real OpenJev classification."""
    probabilities = {label.value: 0.01 for label in Intent}
    probabilities[intent.value] = 0.96
    return {
        "model": "openjev-mlx-4bit",
        "usage": {"input_tokens": 100, "output_tokens": 10},
        "answers": {
            "intent": {
                "type": "choice",
                "choice": intent.value,
                "confidence": confidence,
                "probabilities": probabilities,
            }
        },
    }


@pytest.fixture
def api(monkeypatch):
    """Test doubles only at the HTTP boundary; FastAPI and the SDK remain real."""
    settings = main.Settings(_env_file=None)
    monkeypatch.setattr(main, "Settings", lambda: settings)
    doubles = {
        "jev_body": choice_payload(Intent.ARITHMETIC),
        "jev_status": 200,
        "jev_error": None,
        "ollama_body": {"message": {"content": "Hello, world."}},
        "ollama_status": 200,
    }
    calls = {"jev": [], "ollama": []}

    def jev_http(request):
        assert request.url.path == "/v1/systemone"
        assert request.url.host == "127.0.0.1"
        assert request.headers["authorization"] == "Bearer test-only-key"
        calls["jev"].append(json.loads(request.content))
        if doubles["jev_error"]:
            raise doubles["jev_error"]
        return httpx2.Response(doubles["jev_status"], json=doubles["jev_body"])

    def ollama_http(request):
        assert request.url.path == "/api/chat"
        calls["ollama"].append(json.loads(request.content))
        return httpx.Response(doubles["ollama_status"], json=doubles["ollama_body"])

    with TestClient(main.app) as client:
        sdk = AsyncTypeSafeClient(
            api_key="test-only-key",
            model="openjev",
            base_url="http://127.0.0.1:3000",
            retry=RetryPolicy(max_retries=0),
            transport=httpx2.MockTransport(jev_http),
        )
        ollama = httpx.AsyncClient(
            base_url="http://ollama.test", transport=httpx.MockTransport(ollama_http)
        )
        main.app.state.jev = sdk
        main.app.state.ollama = ollama
        try:
            yield client, doubles, calls
        finally:
            client.portal.call(sdk.aclose)
            client.portal.call(ollama.aclose)
