# OpenJev Agent Router

Minimal local proof of concept: FastAPI receives a request, OpenJev selects a route,
and Python invokes its agent. No TypeSafe account or paid API is used.
Start with `app/jev.py`: it contains the `Choice` question, the five options, and the single model call.

```text
POST /classify -> OpenJev (MLX, port 3000) -> decision + probabilities + confidence
POST /ask     -> OpenJev -> threshold 0.80 -> agent -> result
```

| Route selected by OpenJev | Agent |
| --- | --- |
| TRANSLATE_ES_EN | Translation with a local Ollama model |
| ARITHMETIC | Calculation using Python's AST, without eval |
| WEEKDAY | Date handling with datetime.date |
| OUT_OF_SCOPE | No agent |
| NEEDS_CLARIFICATION | Requests more information, without invoking an agent |

A decision below the threshold also prevents agent execution; the response preserves the original decision.
If an agent receives invalid input, it returns `needs_clarification` even if OpenJev was confident.

## Running

Requires Python 3.12+, an Apple Silicon Mac, and sufficient memory for the 4-bit model.
The published weights occupy approximately 15 GB; this workspace was prepared on a 48 GB Mac.
FastAPI uses `.venv`; the model runtime is isolated in `.openjev-venv`.
For a fresh installation, download the pinned official helpers and weights:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
bash scripts/setup_openjev.sh
```

The setup script verifies published model checksums. Downloads, runtimes, and vendor files
are excluded from version control. `.env` is ready here; `.env.example` documents the settings.
No API key is required. Start OpenJev in one terminal, then FastAPI in another:

```bash
bash scripts/start_openjev.sh
```

```bash
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open [Swagger](http://127.0.0.1:8000/docs) to try the API.
`GET /health` checks FastAPI liveness only, not model readiness.
`GET http://127.0.0.1:3000/v1/version` reports the loaded model and readout configuration.
Classification returns 503 if OpenJev is unavailable; no fallback agent is executed.

Only translation requires Ollama to be running with the model configured in `.env`:

```bash
ollama serve
```

In another terminal, if that model has not been downloaded yet:

```bash
ollama pull llama3.1:8b
```

## Testing OpenJev Separately

```bash
curl -s http://127.0.0.1:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"message":"Calculate: 12 * (3 + 2)"}'
```

The response includes `intent`, `confidence`, `probabilities`, `model`, and `latency_ms`.
These are OpenJev's values, without a generated explanation or agent execution.

## Testing The Orchestrator

```bash
curl -s http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"message":"Calculate: 12 * (3 + 2)"}'
```

```bash
curl -s http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"message":"What day of the week was 2024-01-01?"}'
```

```bash
curl -s http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"message":"Translate into English: Buenos dias, como estas?"}'
```

Also try `Write a poem`, `Help me`, and
`Translate hola into English and calculate 2 + 2` to see cases that do not invoke an agent.

## Reading The Code

1. `app/models.py`: the five intents and the request and response models.
2. `app/jev.py`: `state` is the request; `questions` contains a `Choice` with criteria.
3. `app/main.py`: compares `confidence` against the threshold and executes the selected branch.
4. `app/agents.py`: three small functions that perform the tasks.


