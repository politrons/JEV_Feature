import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from time import perf_counter
from typing import Annotated, Literal
from uuid import uuid4

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeAPIConnectionError, TypeSafeError

from app.agents import (
    AgentInputError,
    AgentUnavailableError,
    arithmetic_agent,
    translation_agent,
    weekday_agent,
)
from app.jev import classify
from app.logging_config import configure_logging, request_id
from app.models import AskRequest, AskResponse, Decision, Intent

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    openjev_base_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:3000")
    openjev_model: str = "openjev"
    jev_confidence_threshold: float = Field(default=0.80, ge=0, le=1, allow_inf_nan=False)
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:8b"
    app_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("openjev_base_url")
    @classmethod
    def require_local_openjev(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.host not in {"127.0.0.1", "localhost", "[::1]"}:
            raise ValueError("OpenJev must use a loopback address for this local POC.")
        return value


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    configure_logging(settings.app_log_level)
    logger.info(
        "Application starting openjev_model=%s confidence_threshold=%.2f ollama_model=%s",
        settings.openjev_model,
        settings.jev_confidence_threshold,
        settings.ollama_model,
    )
    async with AsyncExitStack() as stack:
        app.state.jev = await stack.enter_async_context(
            AsyncTypeSafeClient(
                # The compatible SDK requires a nonempty key; no cloud credential is used.
                api_key="local-openjev",
                base_url=str(settings.openjev_base_url),
                model=settings.openjev_model,
                timeout=60,
                retry=RetryPolicy(max_retries=0),
            )
        )
        logger.info(
            "OpenJev client initialized base_url=%s timeout_seconds=60 retries=0",
            settings.openjev_base_url,
        )
        app.state.ollama = await stack.enter_async_context(
            httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=60)
        )
        app.state.settings = settings
        logger.info("Application ready ollama_timeout_seconds=60")
        try:
            yield
        finally:
            logger.info("Application shutdown started")
    logger.info("Application shutdown completed")


app = FastAPI(title="OpenJev Agent Router", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def trace_request(request: Request, call_next):
    identifier = uuid4().hex
    token = request_id.set(identifier)
    started = perf_counter()
    logger.info("Request received method=%s path=%r", request.method, request.url.path)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = identifier
        log = logger.warning if response.status_code >= 400 else logger.info
        log(
            "Request completed status_code=%d duration_ms=%.2f",
            response.status_code,
            (perf_counter() - started) * 1000,
        )
        return response
    except Exception as exc:
        logger.error(
            "Request failed status_code=500 error_type=%s duration_ms=%.2f",
            type(exc).__name__,
            (perf_counter() - started) * 1000,
        )
        raise
    finally:
        request_id.reset(token)


def get_jev(request: Request) -> AsyncTypeSafeClient:
    return request.app.state.jev


JevClient = Annotated[AsyncTypeSafeClient, Depends(get_jev)]


async def make_decision(client: AsyncTypeSafeClient, message: str) -> Decision:
    try:
        return await classify(client, message)
    except TypeSafeAPIConnectionError as exc:
        logger.error("OpenJev unavailable error_type=%s", type(exc).__name__)
        raise HTTPException(
            503, "OpenJev is unavailable. Start it with bash scripts/start_openjev.sh."
        ) from exc
    except TypeSafeError as exc:
        logger.error(
            "OpenJev classification failed error_type=%s upstream_status=%s",
            type(exc).__name__,
            getattr(exc, "status", None),
        )
        raise HTTPException(
            502, "The local OpenJev server rejected the request or returned an invalid response."
        ) from exc
    except (KeyError, ValueError) as exc:
        logger.error(
            "OpenJev decision rejected reason=invalid_response error_type=%s", type(exc).__name__
        )
        raise HTTPException(502, "OpenJev returned an invalid decision.") from exc


@app.get("/health")
async def health() -> dict[str, bool]:
    logger.info("Application liveness checked")
    return {"ok": True}


@app.post("/classify", response_model=Decision)
async def classify_request(body: AskRequest, client: JevClient) -> Decision:
    decision = await make_decision(client, body.message)
    logger.info("Classification returned intent=%s agent_execution=False", decision.intent.value)
    return decision


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request, client: JevClient) -> AskResponse:
    decision = await make_decision(client, body.message)
    settings: Settings = request.app.state.settings
    logger.info(
        "Routing decision evaluated intent=%s confidence=%.4f threshold=%.4f",
        decision.intent.value,
        decision.confidence,
        settings.jev_confidence_threshold,
    )
    if decision.confidence < settings.jev_confidence_threshold:
        logger.warning("Routing stopped reason=low_confidence status=needs_clarification")
        return AskResponse(
            status="needs_clarification",
            decision=decision,
            message="Confidence is low; specify a single task.",
        )
    if decision.intent == Intent.NEEDS_CLARIFICATION:
        logger.info("Routing stopped reason=missing_or_ambiguous_input status=needs_clarification")
        return AskResponse(
            status="needs_clarification",
            decision=decision,
            message="Specify the task and provide the required input.",
        )
    if decision.intent == Intent.OUT_OF_SCOPE:
        logger.info("Routing stopped reason=no_matching_agent status=out_of_scope")
        return AskResponse(
            status="out_of_scope", decision=decision, message="No agent is available for that task."
        )

    logger.info("Dispatching request agent=%s", decision.intent.value)
    started = perf_counter()
    try:
        match decision.intent:
            case Intent.TRANSLATE_ES_EN:
                result = await translation_agent(
                    request.app.state.ollama, settings.ollama_model, body.message
                )
            case Intent.ARITHMETIC:
                result = arithmetic_agent(body.message)
            case Intent.WEEKDAY:
                result = weekday_agent(body.message)
    except AgentInputError as exc:
        logger.warning(
            "Agent input rejected agent=%s status=needs_clarification duration_ms=%.2f",
            decision.intent.value,
            (perf_counter() - started) * 1000,
        )
        return AskResponse(
            status="needs_clarification", decision=decision, agent=decision.intent, message=str(exc)
        )
    except AgentUnavailableError as exc:
        logger.error(
            "Agent execution failed agent=%s status_code=502 duration_ms=%.2f",
            decision.intent.value,
            (perf_counter() - started) * 1000,
        )
        raise HTTPException(502, str(exc)) from exc
    logger.info(
        "Agent execution completed agent=%s status=completed duration_ms=%.2f",
        decision.intent.value,
        (perf_counter() - started) * 1000,
    )
    return AskResponse(status="completed", decision=decision, agent=decision.intent, result=result)
