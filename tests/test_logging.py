import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.models import Intent
from tests.conftest import choice_payload


def request_messages(caplog, identifier):
    return [
        record.getMessage()
        for record in caplog.records
        if record.name.startswith("app.") and getattr(record, "request_id", None) == identifier
    ]


@pytest.mark.parametrize(
    ("intent", "confidence", "expected_event"),
    [
        (Intent.ARITHMETIC, 0.95, "Agent execution completed agent=ARITHMETIC"),
        (Intent.WEEKDAY, 0.95, "Agent execution completed agent=WEEKDAY"),
        (Intent.TRANSLATE_ES_EN, 0.95, "Agent execution completed agent=TRANSLATE_ES_EN"),
        (Intent.ARITHMETIC, 0.30, "Routing stopped reason=low_confidence"),
        (Intent.NEEDS_CLARIFICATION, 0.95, "Routing stopped reason=missing_or_ambiguous_input"),
        (Intent.OUT_OF_SCOPE, 0.95, "Routing stopped reason=no_matching_agent"),
    ],
)
def test_request_logs_follow_each_routing_outcome(api, caplog, intent, confidence, expected_event):
    caplog.set_level(logging.INFO, logger="app")
    client, doubles, _ = api
    doubles["jev_body"] = choice_payload(intent, confidence)
    message = "Date: 2024-01-01" if intent == Intent.WEEKDAY else "Calculate: 2 + 2"
    response = client.post("/ask", json={"message": message})
    messages = request_messages(caplog, response.headers["X-Request-ID"])
    assert messages[0] == "Request received method=POST path='/ask'"
    assert any("OpenJev classification started" in event for event in messages)
    assert any(
        "OpenJev classification completed" in event and "probabilities=" in event
        for event in messages
    )
    assert any(
        "Routing decision evaluated" in event and "threshold=0.8000" in event for event in messages
    )
    assert any(expected_event in event for event in messages)
    assert messages[-1].startswith("Request completed status_code=200 duration_ms=")


def test_concurrent_requests_keep_separate_log_identifiers(api, caplog):
    caplog.set_level(logging.INFO, logger="app")
    client, _, _ = api

    def send_request(expression):
        return client.post("/ask", json={"message": expression})

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(send_request, ["2 + 2", "3 * 3"]))
    identifiers = {response.headers["X-Request-ID"] for response in responses}
    assert len(identifiers) == 2
    assert {response.json()["result"] for response in responses} == {4, 9}
    for identifier in identifiers:
        messages = request_messages(caplog, identifier)
        assert sum(event.startswith("Request received") for event in messages) == 1
        assert sum(event.startswith("Request completed") for event in messages) == 1
        assert sum(event.startswith("Agent execution completed") for event in messages) == 1


def test_translation_logs_do_not_expose_payloads_or_api_keys(api, caplog):
    caplog.set_level(logging.DEBUG, logger="app")
    client, doubles, _ = api
    doubles["jev_body"] = choice_payload(Intent.TRANSLATE_ES_EN)
    doubles["ollama_body"] = {"message": {"content": "private-translation-output"}}
    response = client.post("/ask", json={"message": "private-request-content"})
    messages = request_messages(caplog, response.headers["X-Request-ID"])
    joined = "\n".join(messages)
    assert "Sending translation request to Ollama" in joined
    assert "Ollama response received status_code=200" in joined
    assert "Translation agent completed output_chars=" in joined
    for secret in ("private-request-content", "private-translation-output", "test-only-key"):
        assert secret not in caplog.text


def test_jev_error_logs_status_without_upstream_error_body(api, caplog):
    caplog.set_level(logging.INFO, logger="app")
    client, doubles, _ = api
    doubles["jev_status"] = 401
    doubles["jev_body"] = {"error": "private-upstream-error"}
    response = client.post("/ask", json={"message": "2 + 2"})
    messages = request_messages(caplog, response.headers["X-Request-ID"])
    assert response.status_code == 502
    assert any(
        "OpenJev classification failed" in event and "upstream_status=401" in event
        for event in messages
    )
    assert not any("Dispatching request" in event for event in messages)
    assert "private-upstream-error" not in caplog.text


@pytest.mark.parametrize(
    ("intent", "message", "expected_event", "status_code"),
    [
        (Intent.ARITHMETIC, "1 / 0", "Agent input rejected agent=ARITHMETIC", 200),
        (Intent.WEEKDAY, "2024-02-30", "Agent input rejected agent=WEEKDAY", 200),
        (
            Intent.TRANSLATE_ES_EN,
            "Translate: hola",
            "Agent execution failed agent=TRANSLATE_ES_EN",
            502,
        ),
    ],
)
def test_agent_failures_are_visible_in_request_logs(
    api, caplog, intent, message, expected_event, status_code
):
    caplog.set_level(logging.INFO, logger="app")
    client, doubles, _ = api
    doubles["jev_body"] = choice_payload(intent)
    if intent == Intent.TRANSLATE_ES_EN:
        doubles["ollama_status"] = 500
        doubles["ollama_body"] = {"error": "private-ollama-error"}
    response = client.post("/ask", json={"message": message})
    assert response.status_code == status_code
    messages = request_messages(caplog, response.headers["X-Request-ID"])
    assert any(expected_event in event for event in messages)
    assert not any("Agent execution completed" in event for event in messages)
    assert "private-ollama-error" not in caplog.text
