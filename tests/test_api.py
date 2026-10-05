import httpx2
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import main
from app.models import Intent
from tests.conftest import choice_payload


def test_classify_exposes_jev_decision_without_executing_an_agent(api):
    client, _, calls = api
    response = client.post("/classify", json={"message": "Calculate: 2 + 2"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "ARITHMETIC"
    assert data["confidence"] == 0.95
    assert data["model"] == "openjev-mlx-4bit"
    assert set(data["probabilities"]) == {intent.value for intent in Intent}
    assert data["latency_ms"] >= 0
    assert calls["ollama"] == []
    request = calls["jev"][0]
    assert request["state"] == {"request": "Calculate: 2 + 2"}
    assert request["model"] == "openjev"
    assert request["questions"]["intent"]["type"] == "choice"
    assert set(request["questions"]["intent"]["criteria"]) == {label.value for label in Intent}


@pytest.mark.parametrize(
    ("intent", "message", "expected"),
    [
        (Intent.ARITHMETIC, "Calculate: 12 * (3 + 2)", 60),
        (Intent.WEEKDAY, "What day of the week was 2024-01-01?", "Monday"),
        (Intent.TRANSLATE_ES_EN, "Translate into English: Hola, mundo.", "Hello, world."),
    ],
)
def test_routes_to_the_selected_agent(api, intent, message, expected):
    client, doubles, calls = api
    doubles["jev_body"] = choice_payload(intent)
    response = client.post("/ask", json={"message": message})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["agent"] == intent.value
    assert data["result"] == expected
    assert len(calls["jev"]) == 1
    assert len(calls["ollama"]) == (intent == Intent.TRANSLATE_ES_EN)
    if calls["ollama"]:
        request = calls["ollama"][0]
        assert request["stream"] is False
        assert request["messages"][1]["content"] == message


@pytest.mark.parametrize("confidence", [0.0, 0.79, 0.7999])
def test_low_confidence_never_executes_an_agent(api, confidence):
    client, doubles, calls = api
    doubles["jev_body"] = choice_payload(Intent.TRANSLATE_ES_EN, confidence)
    response = client.post("/ask", json={"message": "Translate into English: Hola."})
    assert response.json()["status"] == "needs_clarification"
    assert response.json()["agent"] is None
    assert response.json()["result"] is None
    assert response.json()["message"] == "Confidence is low; specify a single task."
    assert calls["ollama"] == []


def test_threshold_is_inclusive(api):
    client, doubles, _ = api
    doubles["jev_body"] = choice_payload(Intent.ARITHMETIC, 0.80)
    assert client.post("/ask", json={"message": "2 + 2"}).json()["result"] == 4


@pytest.mark.parametrize(
    ("intent", "status", "expected_message"),
    [
        (Intent.OUT_OF_SCOPE, "out_of_scope", "No agent is available for that task."),
        (
            Intent.NEEDS_CLARIFICATION,
            "needs_clarification",
            "Specify the task and provide the required input.",
        ),
    ],
)
def test_non_actionable_intents_do_not_execute_an_agent(api, intent, status, expected_message):
    client, doubles, calls = api
    doubles["jev_body"] = choice_payload(intent)
    response = client.post("/ask", json={"message": "Help me."})
    assert response.json()["status"] == status
    assert response.json()["agent"] is None
    assert response.json()["message"] == expected_message
    assert calls["ollama"] == []


@pytest.mark.parametrize(
    ("intent", "message", "expected_message"),
    [
        (
            Intent.ARITHMETIC,
            "Calculate: 1 / 0",
            "Use a valid expression, for example: Calculate: 12 * (3 + 2).",
        ),
        (Intent.WEEKDAY, "Date: 2024-02-30", "The specified date does not exist."),
    ],
)
def test_agent_validates_input_even_when_jev_is_confident(api, intent, message, expected_message):
    client, doubles, _ = api
    doubles["jev_body"] = choice_payload(intent)
    data = client.post("/ask", json={"message": message}).json()
    assert data["status"] == "needs_clarification"
    assert data["result"] is None
    assert data["message"] == expected_message


def test_jev_failure_does_not_trigger_fallback_execution(api):
    client, doubles, calls = api
    doubles["jev_status"] = 401
    doubles["jev_body"] = {"error": "Unauthorized"}
    response = client.post("/ask", json={"message": "2 + 2"})
    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The local OpenJev server rejected the request or returned an invalid response."
    )
    assert calls["ollama"] == []


@pytest.mark.parametrize("bad_answer", ["unknown_label", "missing_confidence", "missing_intent"])
def test_invalid_jev_responses_are_rejected(api, bad_answer):
    client, doubles, calls = api
    answer = doubles["jev_body"]["answers"]["intent"]
    if bad_answer == "unknown_label":
        answer["choice"] = "EXECUTE_SHELL"
    elif bad_answer == "missing_confidence":
        del answer["confidence"]
    else:
        doubles["jev_body"]["answers"] = {}
    assert client.post("/ask", json={"message": "2 + 2"}).status_code == 502
    assert calls["ollama"] == []


@pytest.mark.parametrize("body", [{"message": "   "}, {"message": "x" * 4001}, {}])
def test_invalid_requests_are_rejected_before_jev(api, body):
    client, _, calls = api
    assert client.post("/ask", json=body).status_code == 422
    assert calls["jev"] == []


@pytest.mark.parametrize("payload", [{}, {"message": {"content": " "}}, {"message": None}])
def test_malformed_ollama_response_is_reported(api, payload):
    client, doubles, _ = api
    doubles["jev_body"] = choice_payload(Intent.TRANSLATE_ES_EN)
    doubles["ollama_body"] = payload
    response = client.post("/ask", json={"message": "Translate into English: Hola."})
    assert response.status_code == 502
    assert response.json()["detail"] == "Ollama is unavailable or did not return a translation."


def test_local_client_and_docs_do_not_require_a_cloud_key(monkeypatch):
    settings = main.Settings(_env_file=None)
    monkeypatch.setattr(main, "Settings", lambda: settings)
    with TestClient(main.app) as client:
        assert client.get("/health").json() == {"ok": True}
        assert client.get("/docs").status_code == 200
        assert main.app.state.jev is not None


@pytest.mark.parametrize("error", [httpx2.ConnectError, httpx2.ReadTimeout])
def test_unavailable_openjev_does_not_execute_an_agent(api, error):
    client, doubles, calls = api
    doubles["jev_error"] = error("Test double: local OpenJev is unavailable.")
    response = client.post("/ask", json={"message": "2 + 2"})
    assert response.status_code == 503
    assert response.json()["detail"] == (
        "OpenJev is unavailable. Start it with bash scripts/start_openjev.sh."
    )
    assert calls["ollama"] == []


def test_cloud_endpoint_is_rejected():
    with pytest.raises(ValidationError, match="loopback address"):
        main.Settings(_env_file=None, openjev_base_url="https://api.typesafe.ai")
